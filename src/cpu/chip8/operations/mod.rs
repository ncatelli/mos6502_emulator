use crate::address_map::SafeAddressable;
use crate::cpu::chip8::{
    self,
    microcode::*,
    register::{self, GpRegisters},
    u12::u12,
    Chip8, Display, GenerateRandom,
};
use crate::cpu::Generate;
use crate::prelude::v1::Register;
use parcel::prelude::v1::*;
use std::convert::TryFrom;

pub mod addressing_mode;

/// A placeholder constant error string until a u4 type is implemented. Other
/// assertions are in place so that this should never be encountered.
const NIBBLE_OVERFLOW: &str = "unreachable nibble should be limited to u4.";

/// Represents a mask to binary and against a u8 to return the upper nibble.
const UPPER_NIBBLE_MASK: u8 = 0xf0;

/// Represents a mask to binary and against a u8 to return the lower nibble.
const LOWER_NIBBLE_MASK: u8 = 0x0f;

/// Returns a u8 representing the input byte with the most significant
/// masked limiting the maximum value to 0x0f.
const fn least_significant_nibble_from_u8(x: u8) -> u8 {
    x & LOWER_NIBBLE_MASK
}

/// Generates a u8 from two nibbles. This expectes both input values to
/// respect the maximum value range of a nibble as the most significant bits
/// are left shifted to accommodate the least significant bits.
const fn u8_from_nibbles(msb: u8, lsb: u8) -> u8 {
    let masked_lsb = least_significant_nibble_from_u8(lsb);
    (msb << 4) | masked_lsb
}

#[cfg(test)]
mod tests;

/// ToNibble provides methods for fetching the upper and lower nibble of a byte.
pub trait ToNibble {
    fn to_upper_nibble(&self) -> u8;
    fn to_lower_nibble(&self) -> u8;
}

impl ToNibble for u8 {
    fn to_upper_nibble(&self) -> u8 {
        (self & UPPER_NIBBLE_MASK) >> 4
    }

    fn to_lower_nibble(&self) -> u8 {
        self & LOWER_NIBBLE_MASK
    }
}

/// ToNibbles defines a trait for converting a type from a value into its
/// corresponding nibbles.
pub trait ToNibbleBytes {
    fn to_be_nibbles(&self) -> [u8; 2];
    fn to_le_nibbles(&self) -> [u8; 2];
}

impl ToNibbleBytes for u8 {
    fn to_be_nibbles(&self) -> [u8; 2] {
        [self.to_upper_nibble(), self.to_lower_nibble()]
    }

    fn to_le_nibbles(&self) -> [u8; 2] {
        [self.to_lower_nibble(), self.to_upper_nibble()]
    }
}

/// OpcodeVariant represents all valid instructions with a mapping to their
/// corresponding concrete type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Opcode {
    Cls,
    Ret,
    Call(u12),
    JpNonV0Indexed(u12),
    JpV0Indexed(u12),
    LdAbsolute(u12),
    LdImmediate(GpRegisters, u8),
    LdVxVy(GpRegisters, GpRegisters),
    LdSoundTimerDestTx(GpRegisters),
    LdDelayTimerDestTx(GpRegisters),
    LdDelayTimerSrcTx(GpRegisters),
    LdBcd(GpRegisters),
    LdK(GpRegisters),
    AddImmediate(GpRegisters, u8),
    AddIRegisterIndexed(GpRegisters),
    AddVxVy(GpRegisters, GpRegisters),
    Sub(GpRegisters, GpRegisters),
    Subn(GpRegisters, GpRegisters),
    And(GpRegisters, GpRegisters),
    Or(GpRegisters, GpRegisters),
    Shl(GpRegisters, GpRegisters),
    Shr(GpRegisters, GpRegisters),
    Xor(GpRegisters, GpRegisters),
    SeVxVy(GpRegisters, GpRegisters),
    SeImmediate(GpRegisters, u8),
    SneVxVy(GpRegisters, GpRegisters),
    SneImmediate(GpRegisters, u8),
    ReadRegistersFromMemory(GpRegisters),
    StoreRegistersToMemory(GpRegisters),
    Skp(GpRegisters),
    Sknp(GpRegisters),
    Rnd(GpRegisters, u8),
}

impl<R> Generate<Chip8<R>> for Opcode
where
    R: GenerateRandom<u8>,
{
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        use addressing_mode::*;
        match self {
            Opcode::Cls => Cls.generate(cpu),
            Opcode::Ret => Ret.generate(cpu),
            Opcode::Call(abs) => Call::new(*abs).generate(cpu),
            Opcode::JpNonV0Indexed(abs) => Jp::<NonV0Indexed>::new(*abs).generate(cpu),
            Opcode::JpV0Indexed(abs) => Jp::<V0Indexed>::new(*abs).generate(cpu),
            Opcode::LdAbsolute(abs) => Ld::new(Absolute::new(*abs)).generate(cpu),
            Opcode::LdImmediate(dest, value) => {
                Ld::new(Immediate::new(*dest, *value)).generate(cpu)
            }
            Opcode::LdVxVy(dest, src) => Ld::new(VxVy::new(*src, *dest)).generate(cpu),
            Opcode::LdSoundTimerDestTx(dest) => Ld::new(SoundTimerDestTx::new(*dest)).generate(cpu),
            Opcode::LdDelayTimerDestTx(dest) => Ld::new(DelayTimerDestTx::new(*dest)).generate(cpu),
            Opcode::LdDelayTimerSrcTx(src) => Ld::new(DelayTimerSrcTx::new(*src)).generate(cpu),
            Opcode::LdBcd(reg) => LdBcd::new(VxIIndirect::new(*reg)).generate(cpu),
            Opcode::LdK(dest) => LdK::new(*dest).generate(cpu),
            Opcode::AddImmediate(dest, value) => {
                Add::new(Immediate::new(*dest, *value)).generate(cpu)
            }
            Opcode::AddIRegisterIndexed(reg) => Add::new(IRegisterIndexed::new(*reg)).generate(cpu),
            Opcode::AddVxVy(dest, src) => Add::new(VxVy::new(*src, *dest)).generate(cpu),
            Opcode::Sub(dest, src) => Sub::new(VxVy::new(*src, *dest)).generate(cpu),
            Opcode::Subn(dest, src) => Subn::new(VxVy::new(*src, *dest)).generate(cpu),
            Opcode::And(dest, src) => And::new(VxVy::new(*src, *dest)).generate(cpu),
            Opcode::Or(dest, src) => Or::new(VxVy::new(*src, *dest)).generate(cpu),
            Opcode::Shl(dest, src) => Shl::new(VxVy::new(*src, *dest)).generate(cpu),
            Opcode::Shr(dest, src) => Shr::new(VxVy::new(*src, *dest)).generate(cpu),
            Opcode::Xor(dest, src) => Xor::new(VxVy::new(*src, *dest)).generate(cpu),
            Opcode::SeVxVy(dest, src) => Se::new(VxVy::new(*src, *dest)).generate(cpu),
            Opcode::SeImmediate(reg, value) => Se::new(Immediate::new(*reg, *value)).generate(cpu),
            Opcode::SneVxVy(dest, src) => Sne::new(VxVy::new(*src, *dest)).generate(cpu),
            Opcode::SneImmediate(reg, value) => {
                Sne::new(Immediate::new(*reg, *value)).generate(cpu)
            }
            Opcode::ReadRegistersFromMemory(reg) => {
                ReadRegistersFromMemory::new(VxIIndirect::new(*reg)).generate(cpu)
            }
            Opcode::StoreRegistersToMemory(reg) => {
                StoreRegistersToMemory::new(VxIIndirect::new(*reg)).generate(cpu)
            }
            Opcode::Skp(reg) => Skp::new(*reg).generate(cpu),
            Opcode::Sknp(reg) => Sknp::new(*reg).generate(cpu),
            Opcode::Rnd(reg, value) => Rnd::new(Immediate::new(*reg, *value)).generate(cpu),
        }
    }
}

/// Provides a Parser type for the OpcodeVariant enum. Constructing an
/// OpcodeVariant from a stream of bytes.
pub struct OpcodeVariantParser;

impl<'a> Parser<'a, &'a [(usize, u8)], Opcode> for OpcodeVariantParser {
    #[allow(clippy::type_complexity)]
    fn parse(&self, input: &'a [(usize, u8)]) -> parcel::ParseResult<&'a [(usize, u8)], Opcode> {
        let ms = input.get(0..2).map(|v| [v[0].1, v[1].1]).and_then(|bytes| {
            let [[first, second], [third, fourth]] =
                [u8::to_be_nibbles(&bytes[0]), u8::to_be_nibbles(&bytes[1])];
            let dest_reg: GpRegisters =
                std::convert::TryFrom::<u8>::try_from(second).expect(NIBBLE_OVERFLOW);
            let src_reg: GpRegisters =
                std::convert::TryFrom::<u8>::try_from(third).expect(NIBBLE_OVERFLOW);
            let absolute = u12::from([second, third, fourth]);
            let immediate = u8_from_nibbles(third, fourth);

            match [first, second, third, fourth] {
                [0x0, 0x0, 0xe, 0x0] => Some(Opcode::Cls),
                [0x0, 0x0, 0xe, 0xe] => Some(Opcode::Ret),
                [0x1, _, _, _] => Some(Opcode::JpNonV0Indexed(absolute)),
                [0x2, _, _, _] => Some(Opcode::Call(absolute)),
                [0x3, _, _, _] => Some(Opcode::SeImmediate(dest_reg, immediate)),
                [0x4, _, _, _] => Some(Opcode::SneImmediate(dest_reg, immediate)),
                [0x5, _, _, 0x0] => Some(Opcode::SeVxVy(dest_reg, src_reg)),
                [0x6, _, _, _] => Some(Opcode::LdImmediate(dest_reg, immediate)),
                [0x7, _, _, _] => Some(Opcode::AddImmediate(dest_reg, immediate)),
                [0x8, _, _, 0x0] => Some(Opcode::LdVxVy(dest_reg, src_reg)),
                [0x8, _, _, 0x1] => Some(Opcode::Or(dest_reg, src_reg)),
                [0x8, _, _, 0x2] => Some(Opcode::And(dest_reg, src_reg)),
                [0x8, _, _, 0x3] => Some(Opcode::Xor(dest_reg, src_reg)),
                [0x8, _, _, 0x4] => Some(Opcode::AddVxVy(dest_reg, src_reg)),
                [0x8, _, _, 0x5] => Some(Opcode::Sub(dest_reg, src_reg)),
                [0x8, _, _, 0x6] => Some(Opcode::Shr(dest_reg, src_reg)),
                [0x8, _, _, 0x7] => Some(Opcode::Subn(dest_reg, src_reg)),
                [0x8, _, _, 0xe] => Some(Opcode::Shl(dest_reg, src_reg)),
                [0x9, _, _, 0x0] => Some(Opcode::SneVxVy(dest_reg, src_reg)),
                [0xa, _, _, _] => Some(Opcode::LdAbsolute(absolute)),
                [0xb, _, _, _] => Some(Opcode::JpV0Indexed(absolute)),
                [0xc, _, _, _] => Some(Opcode::Rnd(dest_reg, immediate)),
                [0xe, _, 0x9, 0xe] => Some(Opcode::Skp(dest_reg)),
                [0xe, _, 0xa, 0x1] => Some(Opcode::Sknp(dest_reg)),
                [0xf, _, 0x0, 0x7] => Some(Opcode::LdDelayTimerSrcTx(dest_reg)),
                [0xf, _, 0x0, 0xa] => Some(Opcode::LdK(dest_reg)),
                [0xf, _, 0x1, 0x5] => Some(Opcode::LdDelayTimerDestTx(dest_reg)),
                [0xf, _, 0x1, 0x8] => Some(Opcode::LdSoundTimerDestTx(dest_reg)),
                [0xf, _, 0x1, 0xe] => Some(Opcode::AddIRegisterIndexed(dest_reg)),
                [0xf, _, 0x3, 0x3] => Some(Opcode::LdBcd(dest_reg)),
                [0xf, _, 0x5, 0x5] => Some(Opcode::StoreRegistersToMemory(dest_reg)),
                [0xf, _, 0x6, 0x5] => Some(Opcode::ReadRegistersFromMemory(dest_reg)),
                _ => None,
            }
        });

        // if we get a match, set the appropiate match status headers.
        match ms {
            Some(op) => Ok(MatchStatus::Match {
                /// increment end span to cover non-inclusivity.
                span: input[0].0..(input[1].0 + 1),
                inner: op,
                remainder: &input[2..],
            }),
            None => Ok(MatchStatus::NoMatch(input)),
        }
    }
}

/// Clear the display.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Cls;

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], Cls> for Cls {
    fn parse(&self, input: &'a [(usize, u8)]) -> parcel::ParseResult<&'a [(usize, u8)], Cls> {
        parcel::parsers::byte::expect_bytes(&[0x00, 0xe0])
            .map(|_| Cls::default())
            .parse(input)
    }
}

impl<R> Generate<Chip8<R>> for Cls {
    type Item = Vec<Microcode>;

    fn generate(&self, _: &Chip8<R>) -> Vec<Microcode> {
        vec![Microcode::SetDisplayRange(SetDisplayRange::new(
            (0, 0),
            (Display::x_max(), Display::y_max()),
            false,
        ))]
    }
}

/// Return from a subroutine.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Ret;

impl<R> Generate<Chip8<R>> for Ret {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let current_sp = cpu.sp.read();
        let ret_pc = cpu.stack.read(current_sp as usize);
        let inc_adjusted_addr = ret_pc.wrapping_sub(2);

        vec![
            Microcode::PopStack(PopStack::new(ret_pc)),
            Microcode::Write16bitRegister(Write16bitRegister::new(
                register::WordRegisters::ProgramCounter,
                inc_adjusted_addr,
            )),
        ]
    }
}

/// Reprents a Jp command that is v0 indexed.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct V0Indexed;

/// Reprents an absolute Jp command that is not v0 indexed.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct NonV0Indexed;

/// Jp the associated value to the value of the specified register. Setting
/// the register to the sum.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Jp<T> {
    r#type: std::marker::PhantomData<T>,
    pub address: u12,
}

impl<T> Jp<T> {
    pub fn new(address: u12) -> Self {
        Self {
            r#type: std::marker::PhantomData,
            address,
        }
    }
}

impl<R> Generate<Chip8<R>> for Jp<NonV0Indexed> {
    type Item = Vec<Microcode>;

    fn generate(&self, _: &Chip8<R>) -> Vec<Microcode> {
        vec![Microcode::Write16bitRegister(Write16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            u16::from(self.address).wrapping_sub(2),
        ))]
    }
}

// Jp Absolute + V0

impl<R> Generate<Chip8<R>> for Jp<V0Indexed> {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let v0_val = cpu.read_gp_register(register::GpRegisters::V0);
        let abs_addr = self.address;
        let jmp_addr = abs_addr.wrapping_add(u12::new(v0_val as u16));

        vec![Microcode::Write16bitRegister(Write16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            u16::from(jmp_addr).wrapping_sub(2),
        ))]
    }
}

/// Load the absolute value specified into the I register
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Ld<A> {
    pub addressing_mode: A,
}

impl<A> Ld<A> {
    pub fn new(addressing_mode: A) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for Ld<addressing_mode::Absolute> {
    type Item = Vec<Microcode>;

    fn generate(&self, _: &Chip8<R>) -> Vec<Microcode> {
        vec![Microcode::Write16bitRegister(Write16bitRegister::new(
            register::WordRegisters::I,
            u16::from(self.addressing_mode.addr()),
        ))]
    }
}

impl<R> Generate<Chip8<R>> for Ld<addressing_mode::Immediate> {
    type Item = Vec<Microcode>;

    fn generate(&self, _: &Chip8<R>) -> Vec<Microcode> {
        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(self.addressing_mode.register),
            self.addressing_mode.value,
        ))]
    }
}

impl<R> Generate<Chip8<R>> for Ld<addressing_mode::VxVy> {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let src_val = cpu.read_gp_register(self.addressing_mode.first);

        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(self.addressing_mode.second),
            src_val,
        ))]
    }
}

impl<R> Generate<Chip8<R>> for Ld<addressing_mode::SoundTimerDestTx> {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let src_val = cpu.read_gp_register(self.addressing_mode.src);

        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::TimerRegisters(register::TimerRegisters::Sound),
            src_val,
        ))]
    }
}

impl<R> Generate<Chip8<R>> for Ld<addressing_mode::DelayTimerDestTx> {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let src_val = cpu.read_gp_register(self.addressing_mode.src);

        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::TimerRegisters(register::TimerRegisters::Delay),
            src_val,
        ))]
    }
}

impl<R> Generate<Chip8<R>> for Ld<addressing_mode::DelayTimerSrcTx> {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let src_val = crate::cpu::register::Register::read(&cpu.dt);

        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(self.addressing_mode.dest),
            src_val,
        ))]
    }
}

const fn extract_hundreds_place(x: u8) -> u8 {
    x / 100
}

const fn extract_tens_place(x: u8) -> u8 {
    (x % 100) / 10
}

const fn extract_ones_place(x: u8) -> u8 {
    (x % 100) % 10
}

/// Loads a Binary-Coded Decimal value into the location specified by the
/// addressing mode.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct LdBcd {
    pub addressing_mode: addressing_mode::VxIIndirect,
}

impl LdBcd {
    pub fn new(addressing_mode: addressing_mode::VxIIndirect) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for LdBcd {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let src_val = cpu.read_gp_register(self.addressing_mode.src);
        let hundreds = extract_hundreds_place(src_val);
        let tens = extract_tens_place(src_val);
        let ones = extract_ones_place(src_val);

        vec![
            Microcode::WriteMemory(WriteMemory::new(cpu.i.read(), hundreds)),
            Microcode::WriteMemory(WriteMemory::new(cpu.i.read() + 1, tens)),
            Microcode::WriteMemory(WriteMemory::new(cpu.i.read() + 2, ones)),
        ]
    }
}

/// Wait for a keypress store the corresponding value in the value specified by
/// Vx.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LdK {
    pub dest: register::GpRegisters,
}

impl LdK {
    pub fn new(dest: register::GpRegisters) -> Self {
        Self { dest }
    }
}

impl Default for LdK {
    fn default() -> Self {
        Self::new(register::GpRegisters::V0)
    }
}

impl<R> Generate<Chip8<R>> for LdK {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        match cpu.interrupt {
            // if there is input set, write the input to a register.
            Some(chip8::Interrupt::KeyPress(key_input)) => {
                vec![Microcode::Write8bitRegister(Write8bitRegister::new(
                    register::ByteRegisters::GpRegisters(self.dest),
                    key_input as u8,
                ))]
            }
            // if there is no input, default to looping on this instruction.
            None => vec![Microcode::Dec16bitRegister(Dec16bitRegister::new(
                register::WordRegisters::ProgramCounter,
                2,
            ))],
        }
    }
}

/// Represents the Load Indirect instruction to store a subset of registers at
/// a memory offset defined by the contents of the I register.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct ReadRegistersFromMemory {
    pub addressing_mode: addressing_mode::VxIIndirect,
}

impl ReadRegistersFromMemory {
    pub fn new(addressing_mode: addressing_mode::VxIIndirect) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for ReadRegistersFromMemory {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let reg_inclusive_end_idx = u8::from(self.addressing_mode.src);
        (0..=reg_inclusive_end_idx)
            .into_iter()
            .filter(|idx| *idx <= 0x0f)
            // safe to unwrap due to filter constraint
            .map(|idx| GpRegisters::try_from(idx).unwrap())
            .map(|reg| {
                use crate::address_map::Addressable;
                let i_idx = cpu.i.read() as u16 + reg as u16;
                let i_indirect_val = cpu.address_space.read(i_idx);

                Microcode::Write8bitRegister(Write8bitRegister::new(
                    register::ByteRegisters::GpRegisters(reg),
                    i_indirect_val,
                ))
            })
            .collect()
    }
}

/// Represents the Load Indirect instruction to store a subset of registers at
/// a memory offset defined by the contents of the I register.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct StoreRegistersToMemory {
    pub addressing_mode: addressing_mode::VxIIndirect,
}

impl StoreRegistersToMemory {
    pub fn new(addressing_mode: addressing_mode::VxIIndirect) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for StoreRegistersToMemory {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let reg_inclusive_end_idx = u8::from(self.addressing_mode.src);
        (0..=reg_inclusive_end_idx)
            .into_iter()
            .filter(|idx| *idx <= 0x0f)
            // safe to unwrap due to filter constraint
            .map(|idx| GpRegisters::try_from(idx).unwrap())
            .map(|reg| {
                let src_val = cpu.read_gp_register(reg);
                let i_idx = cpu.i.read() as u16 + reg as u16;
                Microcode::WriteMemory(WriteMemory::new(i_idx, src_val))
            })
            .collect()
    }
}

/// Call subroutine at nnn.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Call {
    pub address: u12,
}

impl Call {
    pub fn new(address: u12) -> Self {
        Self { address }
    }
}

impl<R> Generate<Chip8<R>> for Call {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let current_pc = cpu.pc.read();
        let addr = self.address;
        // decrement 2 to account for PC incrementing.
        let inc_adjusted_addr = u16::from(addr).wrapping_sub(2);

        vec![
            Microcode::PushStack(PushStack::new(current_pc)),
            Microcode::Write16bitRegister(Write16bitRegister::new(
                register::WordRegisters::ProgramCounter,
                inc_adjusted_addr,
            )),
        ]
    }
}

/// Adds the associated value to the value of the specified register. Setting
/// the register to the sum.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Add<A> {
    pub addressing_mode: A,
}

impl<A> Add<A> {
    pub fn new(addressing_mode: A) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for Add<addressing_mode::Immediate> {
    type Item = Vec<Microcode>;

    fn generate(&self, _: &Chip8<R>) -> Vec<Microcode> {
        vec![Microcode::Inc8bitRegister(Inc8bitRegister::new(
            register::ByteRegisters::GpRegisters(self.addressing_mode.register),
            self.addressing_mode.value,
        ))]
    }
}

impl<R> Generate<Chip8<R>> for Add<addressing_mode::IRegisterIndexed> {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let gp_val = cpu.read_gp_register(self.addressing_mode.register);
        vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
            register::WordRegisters::I,
            gp_val as u16,
        ))]
    }
}

impl<R> Generate<Chip8<R>> for Add<addressing_mode::VxVy> {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let src_val = cpu.read_gp_register(self.addressing_mode.first);
        let dest_val = cpu.read_gp_register(self.addressing_mode.second);
        let (result, overflows) = dest_val.overflowing_add(src_val);
        let flag_val = if overflows { 1u8 } else { 0u8 };

        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(self.addressing_mode.second),
                result,
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                flag_val,
            )),
        ]
    }
}

/// Subtracts the associated value from the value of the specified register.
/// Setting the register to the difference. Sets the borrow flag if the
/// difference does not underflow
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Sub {
    pub addressing_mode: addressing_mode::VxVy,
}

impl Sub {
    pub fn new(addressing_mode: addressing_mode::VxVy) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for Sub {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let src_val = cpu.read_gp_register(self.addressing_mode.first);
        let dest_val = cpu.read_gp_register(self.addressing_mode.second);
        let (result, underflows) = dest_val.overflowing_sub(src_val);
        let flag_val = if underflows { 0u8 } else { 1u8 };

        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(self.addressing_mode.second),
                result,
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                flag_val,
            )),
        ]
    }
}

/// Subtracts the associated value from the value of the specified register.
/// Setting the register to the difference. Sets the borrow flag if the
/// difference does not underflow
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Subn {
    pub addressing_mode: addressing_mode::VxVy,
}

impl Subn {
    pub fn new(addressing_mode: addressing_mode::VxVy) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for Subn {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let src_val = cpu.read_gp_register(self.addressing_mode.first);
        let dest_val = cpu.read_gp_register(self.addressing_mode.second);
        let (result, underflows) = src_val.overflowing_sub(dest_val);
        let flag_val = if underflows { 0u8 } else { 1u8 };

        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(self.addressing_mode.second),
                result,
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                flag_val,
            )),
        ]
    }
}

/// And represents a binary & operation.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct And {
    pub addressing_mode: addressing_mode::VxVy,
}

impl And {
    pub fn new(addressing_mode: addressing_mode::VxVy) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for And {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let src_val = cpu.read_gp_register(self.addressing_mode.first);
        let dest_val = cpu.read_gp_register(self.addressing_mode.second);
        let result = dest_val & src_val;

        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(self.addressing_mode.second),
            result,
        ))]
    }
}

/// Or represents a binary | operation.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Or {
    pub addressing_mode: addressing_mode::VxVy,
}

impl Or {
    pub fn new(addressing_mode: addressing_mode::VxVy) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for Or {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let src_val = cpu.read_gp_register(self.addressing_mode.first);
        let dest_val = cpu.read_gp_register(self.addressing_mode.second);
        let result = dest_val | src_val;

        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(self.addressing_mode.second),
            result,
        ))]
    }
}

/// Represents the `SKP Vx` instruction. This instruction skips the next
/// instruction if the value in register `Vx` matches the pressed, if any, key.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Skp {
    register: GpRegisters,
}

impl Skp {
    pub fn new(register: GpRegisters) -> Self {
        Self { register }
    }
}

impl<R> Generate<Chip8<R>> for Skp {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let reg_val = cpu.read_gp_register(self.register);

        match cpu.interrupt {
            Some(chip8::Interrupt::KeyPress(iv)) if iv as u8 == reg_val => {
                vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
                    register::WordRegisters::ProgramCounter,
                    2,
                ))]
            }
            _ => vec![],
        }
    }
}

impl Default for Skp {
    fn default() -> Self {
        Skp::new(GpRegisters::V0)
    }
}

/// Represents the `SKNP Vx` instruction. This instruction skips the next
/// instruction if value in register `Vx` does not match the key, if any,
/// pressed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sknp {
    register: GpRegisters,
}

impl Sknp {
    pub fn new(register: GpRegisters) -> Self {
        Self { register }
    }
}

impl<R> Generate<Chip8<R>> for Sknp {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let reg_val = cpu.read_gp_register(self.register);

        match cpu.interrupt {
            Some(chip8::Interrupt::KeyPress(iv)) if iv as u8 == reg_val => vec![],
            _ => vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
                register::WordRegisters::ProgramCounter,
                2,
            ))],
        }
    }
}

impl Default for Sknp {
    fn default() -> Self {
        Sknp::new(GpRegisters::V0)
    }
}

/// Xor represents a binary ^ operation.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Xor {
    pub addressing_mode: addressing_mode::VxVy,
}

impl Xor {
    pub fn new(addressing_mode: addressing_mode::VxVy) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for Xor {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let src_val = cpu.read_gp_register(self.addressing_mode.first);
        let dest_val = cpu.read_gp_register(self.addressing_mode.second);
        let result = dest_val ^ src_val;

        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(self.addressing_mode.second),
            result,
        ))]
    }
}

/// Shl represents a binary << operation.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Shl {
    pub addressing_mode: addressing_mode::VxVy,
}

impl Shl {
    pub fn new(addressing_mode: addressing_mode::VxVy) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for Shl {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let dest_val = cpu.read_gp_register(self.addressing_mode.second);
        // Set flags to 1 if MSB is 1
        let flags = dest_val >> 7;
        let result = dest_val << 1;

        vec![
            // write overflow if the MSB is 1
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                flags,
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(self.addressing_mode.second),
                result,
            )),
        ]
    }
}

/// Shr represents a binary >> operation.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Shr {
    pub addressing_mode: addressing_mode::VxVy,
}

impl Shr {
    pub fn new(addressing_mode: addressing_mode::VxVy) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for Shr {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let dest_val = cpu.read_gp_register(self.addressing_mode.second);
        // Set flags to 1 if LSB is 1
        let flags = dest_val & 1;
        let result = dest_val >> 1;

        vec![
            // write flags if the LSB is 1
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                flags,
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(self.addressing_mode.second),
                result,
            )),
        ]
    }
}

/// Se skips the next instruction if the operands are equivalent.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Se<A> {
    pub addressing_mode: A,
}

impl<A> Se<A> {
    pub fn new(addressing_mode: A) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for Se<addressing_mode::Immediate> {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let reg_val = cpu.read_gp_register(self.addressing_mode.register);
        let value = self.addressing_mode.value;

        if reg_val == value {
            vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
                register::WordRegisters::ProgramCounter,
                2,
            ))]
        } else {
            vec![]
        }
    }
}

impl<R> Generate<Chip8<R>> for Se<addressing_mode::VxVy> {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let first_reg_val = cpu.read_gp_register(self.addressing_mode.first);
        let second_reg_val = cpu.read_gp_register(self.addressing_mode.second);

        if first_reg_val == second_reg_val {
            vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
                register::WordRegisters::ProgramCounter,
                2,
            ))]
        } else {
            vec![]
        }
    }
}

/// Sne skips the next instruction if the operands are not equivalent.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Sne<A> {
    pub addressing_mode: A,
}

impl<A> Sne<A> {
    pub fn new(addressing_mode: A) -> Self {
        Self { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for Sne<addressing_mode::Immediate> {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let reg_val = cpu.read_gp_register(self.addressing_mode.register);
        let value = self.addressing_mode.value;

        if reg_val != value {
            vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
                register::WordRegisters::ProgramCounter,
                2,
            ))]
        } else {
            vec![]
        }
    }
}

impl<R> Generate<Chip8<R>> for Sne<addressing_mode::VxVy> {
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let first_reg_val = cpu.read_gp_register(self.addressing_mode.first);
        let second_reg_val = cpu.read_gp_register(self.addressing_mode.second);

        if first_reg_val != second_reg_val {
            vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
                register::WordRegisters::ProgramCounter,
                2,
            ))]
        } else {
            vec![]
        }
    }
}

/// Rnd generates a random 8-bit value to be applied against a mask.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Rnd {
    pub addressing_mode: addressing_mode::Immediate,
}

impl Rnd {
    pub fn new(addressing_mode: addressing_mode::Immediate) -> Self {
        Rnd { addressing_mode }
    }
}

impl<R> Generate<Chip8<R>> for Rnd
where
    R: GenerateRandom<u8>,
{
    type Item = Vec<Microcode>;

    fn generate(&self, cpu: &Chip8<R>) -> Vec<Microcode> {
        let rand = cpu.rng.random();
        let value = rand & self.addressing_mode.value;

        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(self.addressing_mode.register),
            value,
        ))]
    }
}
