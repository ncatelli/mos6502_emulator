use crate::address_map::{AddressMap, Addressable};
use crate::cpu::Execute;
use crate::cpu::{register::Register, Cpu, StepState};
use parcel::Parser;

use super::ExecuteMut;

mod memory;
mod microcode;
mod operations;
mod register;
mod u12;

/// Defines a trait for implementing random number generation.
pub trait GenerateRandom<T> {
    /// Random should return a value of a constrained type T on each invocation.
    fn random(&self) -> T;
}

impl<F> GenerateRandom<u8> for F
where
    F: Fn() -> u8,
{
    fn random(&self) -> u8 {
        (self)()
    }
}

/// Generates a random byte using `/dev/random` as the seed for data.
#[derive(Default, Debug, Clone, Copy)]
pub struct UnixRandomNumberGenerator;

impl GenerateRandom<u8> for UnixRandomNumberGenerator {
    fn random(&self) -> u8 {
        use std::fs;
        use std::io::prelude::*;

        let mut buf: [u8; 1] = [0];
        fs::File::open("/dev/random")
            .and_then(|mut f| f.read_exact(&mut buf))
            .expect("cannot read exactly one byte from /dev/random.");

        buf[0]
    }
}

/// Represents an interrupt, example being a keypress.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Interrupt {
    KeyPress(KeyInputValue),
}

/// KeyInputValue represents all valid input keys that may be input from the
/// keyboard.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum KeyInputValue {
    Key0,
    Key1,
    Key2,
    Key3,
    Key4,
    Key5,
    Key6,
    Key7,
    Key8,
    Key9,
    KeyA,
    KeyB,
    KeyC,
    KeyD,
    KeyE,
    KeyF,
}

/// Display mimics the display matrix for the CHIP-8 isa.
#[derive(Debug, Clone, Copy)]
pub struct Display {
    inner: [[bool; 64]; 32],
}

impl Default for Display {
    fn default() -> Self {
        Self {
            inner: [[false; 64]; 32],
        }
    }
}

impl Display {
    /// Returns the maximum number of columns per row.
    pub fn x_max() -> usize {
        64
    }

    /// Returns the maximum number of rows.
    pub fn y_max() -> usize {
        32
    }

    /// gets the value of the pixel specified by the cartesian coordinates `x`,
    /// `y`. If the coordinates are within range, an `Option::Some(bool)` with
    /// the value of the pixel is returned. Otherwise `Option::None` is
    /// returned.
    pub fn pixel(&self, x: usize, y: usize) -> Option<bool> {
        if x < Self::x_max() && y < Self::y_max() {
            Some(self.inner[y][x])
        } else {
            None
        }
    }

    /// sets the value of the pixel specified by the cartesian coordinates `x`,
    /// `y` to the boolean value specified by `pixel_on`. If the coordinates
    /// are within range, a `(Self, Option::Some(bool))` with the previous
    /// value of the pixel is returned. Otherwise `(Self, Option::None)` is
    /// returned.
    pub fn write_pixel(mut self, x: usize, y: usize, pixel_on: bool) -> (Self, Option<bool>) {
        let modified = self.write_pixel_mut(x, y, pixel_on);
        (self, modified)
    }

    /// sets the value of the pixel specified by the cartesian coordinates `x`,
    /// `y` to the boolean value specified by `pixel_on`. If the coordinates
    /// are within range, an `Option::Some(bool)` with the previous value of
    /// the pixel is returned. Otherwise `Option::None` is returned.
    pub fn write_pixel_mut(&mut self, x: usize, y: usize, pixel_on: bool) -> Option<bool> {
        if let Some(previous_value) = self.pixel(x, y) {
            // if we can get the previous value, it's safe to write a new one.
            self.inner[y][x] = pixel_on;
            Some(previous_value)
        } else {
            None
        }
    }
}

/// Represents the address the program counter is set to on chip reset.
const RESET_PC_VECTOR: u16 = 0x200;

/// Chip8 represents a CHIP-8 CPU.
#[derive(Debug, Clone)]
pub struct Chip8<R> {
    stack: memory::Ring<u16>,
    address_space: AddressMap<u16, u8>,
    dt: register::ClockDecrementing,
    st: register::ClockDecrementing,
    pc: register::ProgramCounter,
    sp: register::StackPointer,
    i: register::GeneralPurpose<u16>,
    gp_registers: [register::GeneralPurpose<u8>; 0xf],
    display: Display,
    interrupt: Option<Interrupt>,
    rng: R,
}

impl<R> Chip8<R> {
    pub fn with_pc_register(mut self, reg: register::ProgramCounter) -> Self {
        self.pc = reg;
        self
    }

    pub fn with_sp_register(mut self, reg: register::StackPointer) -> Self {
        self.sp = reg;
        self
    }

    pub fn with_timer_register(
        mut self,
        reg_type: register::TimerRegisters,
        reg: register::ClockDecrementing,
    ) -> Self {
        match reg_type {
            register::TimerRegisters::Sound => self.st = reg,
            register::TimerRegisters::Delay => self.dt = reg,
        };
        self
    }

    pub fn with_i_register(mut self, reg: register::GeneralPurpose<u16>) -> Self {
        self.i = reg;
        self
    }

    pub fn with_gp_register(
        mut self,
        reg_type: register::GpRegisters,
        reg: register::GeneralPurpose<u8>,
    ) -> Self {
        self.gp_registers[reg_type as usize] = reg;
        self
    }

    /// Provides a convenient method for unwrapping a GpRegister enum to a
    /// corresponding read of it's namesake register.
    pub fn read_gp_register(&self, reg: register::GpRegisters) -> u8 {
        self.gp_registers
            .get(reg as usize)
            .map(|cpu_reg| cpu_reg.read())
            // Should never fail due to bounded array.
            .unwrap()
    }

    /// Returns an instance of Chip8 with a new random number generator.
    pub fn with_rng<NR>(self, rng: NR) -> Chip8<NR> {
        Chip8 {
            stack: self.stack,
            address_space: self.address_space,
            dt: self.dt,
            st: self.st,
            pc: self.pc,
            sp: self.sp,
            i: self.i,
            gp_registers: self.gp_registers,
            display: self.display,
            interrupt: self.interrupt,
            rng,
        }
    }

    /// Takes and invokes a function that returns an optional Interrupt.
    /// Returning the newly modified state.
    pub fn with_interrupt<F>(self, f: F) -> Self
    where
        F: Fn() -> Option<Interrupt>,
    {
        Self {
            stack: self.stack,
            address_space: self.address_space,
            dt: self.dt,
            st: self.st,
            pc: self.pc,
            sp: self.sp,
            i: self.i,
            gp_registers: self.gp_registers,
            display: self.display,
            interrupt: (f)(),
            rng: self.rng,
        }
    }

    /// Takes and invokes a function that modies the types display, returning
    ///the newly modified state.
    pub fn with_display<F>(self, f: F) -> Self
    where
        F: Fn(Display) -> Display,
    {
        Self {
            stack: self.stack,
            address_space: self.address_space,
            dt: self.dt,
            st: self.st,
            pc: self.pc,
            sp: self.sp,
            i: self.i,
            gp_registers: self.gp_registers,
            display: (f)(self.display),
            interrupt: None,
            rng: self.rng,
        }
    }
}

impl<R> Chip8<R>
where
    R: Default,
{
    /// Resets a cpu to a clean state.
    pub fn reset() -> Self {
        Self::default()
    }
}

impl<R> Default for Chip8<R>
where
    R: Default,
{
    fn default() -> Self {
        type Rom =
            crate::address_map::memory::Memory<crate::address_map::memory::ReadOnly, u16, u8>;
        type Ram =
            crate::address_map::memory::Memory<crate::address_map::memory::ReadWrite, u16, u8>;

        Self {
            stack: memory::Ring::new(16),
            address_space: AddressMap::default()
                .register(0..=0x1ff, Box::new(Rom::new(0, 0x1ff)))
                .unwrap()
                .register(0x200..=0xfff, Box::new(Ram::new(0x200, 0xfff)))
                .unwrap(),
            dt: register::ClockDecrementing::default(),
            st: register::ClockDecrementing::default(),
            pc: register::ProgramCounter::with_value(RESET_PC_VECTOR),
            sp: register::StackPointer::default(),
            i: register::GeneralPurpose::default(),
            gp_registers: [register::GeneralPurpose::default(); 0xf],
            display: Display::default(),
            interrupt: None,
            rng: <R>::default(),
        }
    }
}

impl<R> Cpu<Chip8<R>> for Chip8<R>
where
    R: 'static + Clone + GenerateRandom<u8>,
{
    fn run(self, cycles: usize) -> StepState<Chip8<R>> {
        let state = self
            .clone()
            .into_iter()
            .take(cycles)
            .flatten()
            .fold(self, |c, mc| mc.execute(c));
        StepState::from(state)
    }
}

impl<R> IntoIterator for Chip8<R>
where
    R: 'static + Clone + GenerateRandom<u8>,
{
    type Item = Vec<microcode::Microcode>;
    type IntoIter = Chip8IntoIterator<R>;

    fn into_iter(self) -> Self::IntoIter {
        Chip8IntoIterator::new(self)
    }
}

pub struct Chip8IntoIterator<R> {
    state: Chip8<R>,
}

impl<R> From<Chip8IntoIterator<R>> for Chip8<R> {
    fn from(src: Chip8IntoIterator<R>) -> Self {
        src.state
    }
}

impl<R> Chip8IntoIterator<R> {
    fn new(state: Chip8<R>) -> Self {
        Chip8IntoIterator { state }
    }
}

impl<R> Iterator for Chip8IntoIterator<R>
where
    R: 'static + Clone + GenerateRandom<u8>,
{
    type Item = Vec<microcode::Microcode>;

    fn next(&mut self) -> Option<Vec<microcode::Microcode>> {
        use crate::cpu::Generate;
        let pc = self.state.pc.read();
        let opcodes: [(usize, u8); 2] = [
            (pc as usize, self.state.address_space.read(pc)),
            ((pc as usize + 1), self.state.address_space.read(pc + 1)),
        ];

        // Parse correct operation
        let ops = match operations::OpcodeVariantParser.parse(&opcodes[..]) {
            Ok(parcel::MatchStatus::Match {
                span: _,
                remainder: _,
                inner: op,
            }) => Ok(op),
            _ => Err(format!(
                "No match found for {:#02x}",
                u16::from_be_bytes([opcodes[0].1, opcodes[1].1])
            )),
        }
        .unwrap();

        let microcode_steps: Vec<microcode::Microcode> = ops
            .generate(&self.state)
            .into_iter()
            .chain(vec![microcode::Microcode::Inc16bitRegister(
                // increment the PC by instruction size.
                microcode::Inc16bitRegister::new(register::WordRegisters::ProgramCounter, 2),
            )])
            .collect();

        self.state = microcode_steps
            .iter()
            .fold(self.state.clone(), |cpu, mc| mc.execute(cpu));

        Some(microcode_steps)
    }
}

// microcode execution

// For any implementation of ExecuteMut<M> for a given CPU Execute is implemented.
impl<M, R> crate::cpu::Execute<Chip8<R>> for M
where
    Chip8<R>: ExecuteMut<M>,
{
    fn execute(self, mut cpu: Chip8<R>) -> Chip8<R> {
        cpu.execute_mut(&self);
        cpu
    }
}

impl<R> crate::cpu::ExecuteMut<microcode::Microcode> for Chip8<R> {
    fn execute_mut(&mut self, mc: &microcode::Microcode) {
        match mc {
            microcode::Microcode::WriteMemory(mc) => self.execute_mut(mc),
            microcode::Microcode::Write8bitRegister(mc) => self.execute_mut(mc),
            microcode::Microcode::Inc8bitRegister(mc) => self.execute_mut(mc),
            microcode::Microcode::Dec8bitRegister(mc) => self.execute_mut(mc),
            microcode::Microcode::Write16bitRegister(mc) => self.execute_mut(mc),
            microcode::Microcode::Inc16bitRegister(mc) => self.execute_mut(mc),
            microcode::Microcode::Dec16bitRegister(mc) => self.execute_mut(mc),
            microcode::Microcode::PushStack(mc) => self.execute_mut(mc),
            microcode::Microcode::PopStack(mc) => self.execute_mut(mc),
            microcode::Microcode::KeyPress(mc) => self.execute_mut(mc),
            microcode::Microcode::KeyRelease => self.execute_mut(&microcode::KeyRelease),
            microcode::Microcode::SetDisplayPixel(mc) => self.execute_mut(mc),
            microcode::Microcode::SetDisplayRange(mc) => self.execute_mut(mc),
        }
    }
}

impl<R> crate::cpu::ExecuteMut<microcode::WriteMemory> for Chip8<R> {
    fn execute_mut(&mut self, mc: &microcode::WriteMemory) {
        self.address_space.write(mc.address, mc.value).unwrap();
    }
}

impl<R> crate::cpu::ExecuteMut<microcode::Write8bitRegister> for Chip8<R> {
    fn execute_mut(&mut self, mc: &microcode::Write8bitRegister) {
        use register::{ByteRegisters, ClockDecrementing, GeneralPurpose, TimerRegisters};

        match mc.register {
            register::ByteRegisters::GpRegisters(gpr) => {
                self.gp_registers[gpr as usize] = GeneralPurpose::with_value(mc.value);
            }
            ByteRegisters::TimerRegisters(TimerRegisters::Delay) => {
                self.dt = ClockDecrementing::with_value(mc.value);
            }
            ByteRegisters::TimerRegisters(TimerRegisters::Sound) => {
                self.st = ClockDecrementing::with_value(mc.value);
            }
        }
    }
}

impl<R> crate::cpu::ExecuteMut<microcode::Inc8bitRegister> for Chip8<R> {
    fn execute_mut(&mut self, mc: &microcode::Inc8bitRegister) {
        use register::{ByteRegisters, ClockDecrementing, GeneralPurpose, TimerRegisters};

        match mc.register {
            register::ByteRegisters::GpRegisters(gpr) => {
                let cpu_reg_value = self.read_gp_register(gpr);
                let new_val = cpu_reg_value.wrapping_add(mc.value);
                self.gp_registers[gpr as usize] = GeneralPurpose::with_value(new_val);
            }
            ByteRegisters::TimerRegisters(TimerRegisters::Delay) => {
                let cpu_reg_value = self.dt.read();
                let new_val = cpu_reg_value.wrapping_add(mc.value);
                self.dt = ClockDecrementing::with_value(new_val);
            }
            ByteRegisters::TimerRegisters(TimerRegisters::Sound) => {
                let cpu_reg_value = self.st.read();
                let new_val = cpu_reg_value.wrapping_add(mc.value);
                self.st = ClockDecrementing::with_value(new_val);
            }
        }
    }
}

impl<R> crate::cpu::ExecuteMut<microcode::Dec8bitRegister> for Chip8<R> {
    fn execute_mut(&mut self, _: &microcode::Dec8bitRegister) {}
}

impl<R> crate::cpu::ExecuteMut<microcode::Write16bitRegister> for Chip8<R> {
    fn execute_mut(&mut self, mc: &microcode::Write16bitRegister) {
        match mc.register {
            register::WordRegisters::I => {
                self.i = register::GeneralPurpose::with_value(mc.value);
            }
            register::WordRegisters::ProgramCounter => {
                self.pc = register::ProgramCounter::with_value(mc.value);
            }
        }
    }
}

impl<R> crate::cpu::ExecuteMut<microcode::Inc16bitRegister> for Chip8<R> {
    fn execute_mut(&mut self, mc: &microcode::Inc16bitRegister) {
        match mc.register {
            register::WordRegisters::I => {
                let i = self.i.read();
                self.i = register::GeneralPurpose::with_value(i.wrapping_add(mc.value));
            }
            register::WordRegisters::ProgramCounter => {
                let pc = self.pc.read();
                self.pc = register::ProgramCounter::with_value(pc.wrapping_add(mc.value));
            }
        }
    }
}

impl<R> crate::cpu::ExecuteMut<microcode::Dec16bitRegister> for Chip8<R> {
    fn execute_mut(&mut self, _: &microcode::Dec16bitRegister) {}
}

impl<R> crate::cpu::ExecuteMut<microcode::PushStack> for Chip8<R> {
    fn execute_mut(&mut self, mc: &microcode::PushStack) {
        // increment stack pointer before doing any data writes.
        let sp = self.sp.read().wrapping_add(1);
        self.sp.write(sp);

        // push the value from mc onto the new stack location.
        // due to the protections applied by the `StackPointer`'s `.read()
        // method this should be fairly safe to unwrap as it can't overflow 16
        // places.
        self.stack.write(self.sp.read() as usize, mc.value).unwrap();
    }
}

impl<R> crate::cpu::ExecuteMut<microcode::PopStack> for Chip8<R> {
    fn execute_mut(&mut self, _: &microcode::PopStack) {
        // decrement stack pointer.
        let sp = self.sp.read().wrapping_sub(1);
        self.sp.write(sp);
    }
}

impl<R> crate::cpu::ExecuteMut<microcode::KeyPress> for Chip8<R> {
    fn execute_mut(&mut self, mc: &microcode::KeyPress) {
        self.interrupt = Some(Interrupt::KeyPress(mc.value));
    }
}

impl<R> crate::cpu::ExecuteMut<microcode::KeyRelease> for Chip8<R> {
    fn execute_mut(&mut self, _: &microcode::KeyRelease) {
        self.interrupt = None;
    }
}

impl<R> crate::cpu::ExecuteMut<microcode::SetDisplayPixel> for Chip8<R> {
    fn execute_mut(&mut self, mc: &microcode::SetDisplayPixel) {
        let microcode::SetDisplayPixel((x, y), pixel_value) = *mc;
        self.display.write_pixel_mut(x, y, pixel_value);
    }
}

impl<R> crate::cpu::ExecuteMut<microcode::SetDisplayRange> for Chip8<R> {
    fn execute_mut(&mut self, mc: &microcode::SetDisplayRange) {
        let microcode::SetDisplayRange {
            start: (start_x, start_y),
            end: (end_x, end_y),
            value: pixel_value,
        } = *mc;

        let start_offset = (start_y * Display::x_max()) + start_x;
        // add 1 to cover non-inclusive range.
        let modified = ((end_y - start_y) * Display::x_max()) + (end_x - start_x) + 1;
        self.display
            .inner
            .iter_mut()
            .flatten()
            .skip(start_offset)
            .take(modified)
            .for_each(|pixel| *pixel = pixel_value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::Cpu;

    #[test]
    fn should_execute_infinitely_on_jump_to_reset_vector() {
        let mut cpu = Chip8::<()>::default().with_rng(|| 0u8);
        cpu.address_space.write(0x200, 0x12).unwrap();
        cpu.address_space.write(0x201, 0x00).unwrap();

        assert_eq!(0x200, cpu.pc.read());
        let state = cpu.run(5).unwrap();
        assert_eq!(0x200, state.pc.read())
    }

    #[test]
    fn should_execute_add_immediate() {
        let mut cpu = Chip8::<()>::default().with_rng(|| 0u8).with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::with_value(0x05),
        );
        cpu.address_space.write(0x200, 0x70).unwrap();
        cpu.address_space.write(0x201, 0xfa).unwrap();

        let state = cpu.run(1).unwrap();
        assert_eq!(0xff, state.read_gp_register(register::GpRegisters::V0))
    }

    #[test]
    fn should_clear_input_idempotently() {
        let cpu = Chip8::<()>::default()
            .with_interrupt(|| Some(Interrupt::KeyPress(KeyInputValue::Key0)));

        assert_eq!(
            Some(Interrupt::KeyPress(KeyInputValue::Key0)),
            cpu.interrupt
        );
        assert_eq!(
            None,
            // clear input multiple times
            cpu.with_interrupt(|| None)
                .with_interrupt(|| None)
                .interrupt
        )
    }

    #[test]
    fn should_set_a_given_pixel_to_a_given_value() {
        let cpu = Chip8::<()>::default().with_display(|d| d.write_pixel(1, 1, true).0);

        assert_eq!(Some(false), cpu.display.pixel(2, 1));
        assert_eq!(Some(true), cpu.display.pixel(1, 1));
        assert_eq!(Some(false), cpu.display.pixel(1, 2));
    }

    #[test]
    fn should_set_a_given_pixel_range_to_a_given_value() {
        let mut cpu = Chip8::<()>::default();

        ExecuteMut::<microcode::SetDisplayRange>::execute_mut(
            &mut cpu,
            &microcode::SetDisplayRange::new((0, 0), (63, 31), true),
        );

        for (pos, &v) in cpu.display.inner.iter().flatten().enumerate() {
            assert_eq!(true, v, "no match for position: {}, {}", pos, v);
        }
    }
}
