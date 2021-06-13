use crate::cpu::chip8::{operations::ToNibbleBytes, register, u12::u12};

/// A placeholder constant error string until a u4 type is implemented. Other
/// assertions are in place so that this should never be encountered.
const NIBBLE_OVERFLOW: &str = "unreachable nibble should be limited to u4.";

pub trait AddressingMode {}

/// Implied represents a type that explicitly implies it's addressing mode through a 2-byte mnemonic code.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Implied;

impl AddressingMode for Implied {}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Absolute(u12);

impl AddressingMode for Absolute {}

impl Absolute {
    #[allow(dead_code)]
    pub fn new(addr: u12) -> Self {
        Self(addr)
    }

    pub fn addr(&self) -> u12 {
        self.0
    }
}

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], Absolute> for Absolute {
    fn parse(&self, input: &'a [(usize, u8)]) -> parcel::ParseResult<&'a [(usize, u8)], Absolute> {
        parcel::take_n(parcel::parsers::byte::any_byte(), 2)
            .map(|bytes| [bytes[0].to_be_nibbles(), bytes[1].to_be_nibbles()])
            .map(|[[_, first], [second, third]]| {
                let upper = 0x0f & first;
                let lower = (second << 4) | third;
                u12::new(u16::from_be_bytes([upper, lower]))
            })
            .map(Absolute)
            .parse(input)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Immediate {
    pub register: register::GpRegisters,
    pub value: u8,
}

impl AddressingMode for Immediate {}

impl Immediate {
    pub fn new(register: register::GpRegisters, value: u8) -> Self {
        Self { register, value }
    }
}

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], Immediate> for Immediate {
    fn parse(&self, input: &'a [(usize, u8)]) -> parcel::ParseResult<&'a [(usize, u8)], Immediate> {
        parcel::take_n(parcel::parsers::byte::any_byte(), 2)
            .map(|bytes| [bytes[0].to_be_nibbles(), bytes[1].to_be_nibbles()])
            .map(|[[_, first], [second, third]]| {
                let upper = 0x0f & first;
                let lower = (second << 4) | third;
                let reg = std::convert::TryFrom::<u8>::try_from(upper).expect(NIBBLE_OVERFLOW);

                (reg, lower)
            })
            .map(|(register, value)| Immediate::new(register, value))
            .parse(input)
    }
}

impl Default for Immediate {
    fn default() -> Self {
        Self {
            register: register::GpRegisters::V0,
            value: 0,
        }
    }
}

/// Represents an operation on the I register indexed by a General-Purpose
/// register.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IRegisterIndexed {
    pub register: register::GpRegisters,
}

impl AddressingMode for IRegisterIndexed {}

impl IRegisterIndexed {
    pub fn new(register: register::GpRegisters) -> Self {
        Self { register }
    }
}

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], IRegisterIndexed> for IRegisterIndexed {
    fn parse(
        &self,
        input: &'a [(usize, u8)],
    ) -> parcel::ParseResult<&'a [(usize, u8)], IRegisterIndexed> {
        parcel::take_n(parcel::parsers::byte::any_byte(), 2)
            .map(|bytes| [bytes[0].to_be_nibbles(), bytes[1].to_be_nibbles()])
            .map(|[[_, first], _]| {
                let upper = 0x0f & first;
                std::convert::TryFrom::<u8>::try_from(upper).expect(NIBBLE_OVERFLOW)
            })
            .map(IRegisterIndexed::new)
            .parse(input)
    }
}

impl Default for IRegisterIndexed {
    fn default() -> Self {
        Self {
            register: register::GpRegisters::V0,
        }
    }
}

/// Represents a register to register general-purpose operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VxVy {
    /// Represents the first register defined in this address mode. Often
    /// times this will represent a destination register.
    ///
    /// # Example
    ///
    /// `<mnemonic> <first> <second>` or `Add <first> <second>`
    pub first: register::GpRegisters,

    /// Represents the second register defined in this address mode. Often
    /// times this will represent a source register.
    ///
    /// # Example
    ///
    /// `<mnemonic> <first> <second>` or `Add <first> <second>`
    pub second: register::GpRegisters,
}

impl AddressingMode for VxVy {}

impl VxVy {
    pub fn new(src: register::GpRegisters, dest: register::GpRegisters) -> Self {
        Self {
            first: src,
            second: dest,
        }
    }
}

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], VxVy> for VxVy {
    fn parse(&self, input: &'a [(usize, u8)]) -> parcel::ParseResult<&'a [(usize, u8)], VxVy> {
        parcel::take_n(parcel::parsers::byte::any_byte(), 2)
            .map(|bytes| [bytes[0].to_be_nibbles(), bytes[1].to_be_nibbles()])
            .map(|[[_, first], [second, _]]| {
                let dest =
                    std::convert::TryFrom::<u8>::try_from(0x0f & first).expect(NIBBLE_OVERFLOW);
                let src =
                    std::convert::TryFrom::<u8>::try_from(0x0f & second).expect(NIBBLE_OVERFLOW);

                (src, dest)
            })
            .map(|(src, dest)| VxVy::new(src, dest))
            .parse(input)
    }
}

impl Default for VxVy {
    fn default() -> Self {
        Self {
            first: register::GpRegisters::V0,
            second: register::GpRegisters::V0,
        }
    }
}

/// Represents a register to register operation transfering a value from a
/// register to the Sound Timer register.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoundTimerDestTx {
    pub src: register::GpRegisters,
}

impl AddressingMode for SoundTimerDestTx {}

impl SoundTimerDestTx {
    pub fn new(src: register::GpRegisters) -> Self {
        Self { src }
    }
}

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], SoundTimerDestTx> for SoundTimerDestTx {
    fn parse(
        &self,
        input: &'a [(usize, u8)],
    ) -> parcel::ParseResult<&'a [(usize, u8)], SoundTimerDestTx> {
        parcel::take_n(parcel::parsers::byte::any_byte(), 2)
            .map(|bytes| [bytes[0].to_be_nibbles(), bytes[1].to_be_nibbles()])
            .map(|[[_, first], _]| {
                std::convert::TryFrom::<u8>::try_from(0x0f & first).expect(NIBBLE_OVERFLOW)
            })
            .map(SoundTimerDestTx::new)
            .parse(input)
    }
}

impl Default for SoundTimerDestTx {
    fn default() -> Self {
        Self {
            src: register::GpRegisters::V0,
        }
    }
}

/// Represents a register to register operation transfering a value from a
/// register to the Delay Timer register.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayTimerDestTx {
    pub src: register::GpRegisters,
}

impl AddressingMode for DelayTimerDestTx {}

impl DelayTimerDestTx {
    pub fn new(src: register::GpRegisters) -> Self {
        Self { src }
    }
}

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], DelayTimerDestTx> for DelayTimerDestTx {
    fn parse(
        &self,
        input: &'a [(usize, u8)],
    ) -> parcel::ParseResult<&'a [(usize, u8)], DelayTimerDestTx> {
        parcel::take_n(parcel::parsers::byte::any_byte(), 2)
            .map(|bytes| [bytes[0].to_be_nibbles(), bytes[1].to_be_nibbles()])
            .map(|[[_, first], _]| {
                std::convert::TryFrom::<u8>::try_from(0x0f & first).expect(NIBBLE_OVERFLOW)
            })
            .map(DelayTimerDestTx::new)
            .parse(input)
    }
}

impl Default for DelayTimerDestTx {
    fn default() -> Self {
        Self {
            src: register::GpRegisters::V0,
        }
    }
}

/// Represents a register to register operation transfering a value from a
/// register to the Delay Timer register.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayTimerSrcTx {
    pub dest: register::GpRegisters,
}

impl AddressingMode for DelayTimerSrcTx {}

impl DelayTimerSrcTx {
    pub fn new(dest: register::GpRegisters) -> Self {
        Self { dest }
    }
}

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], DelayTimerSrcTx> for DelayTimerSrcTx {
    fn parse(
        &self,
        input: &'a [(usize, u8)],
    ) -> parcel::ParseResult<&'a [(usize, u8)], DelayTimerSrcTx> {
        parcel::take_n(parcel::parsers::byte::any_byte(), 2)
            .map(|bytes| [bytes[0].to_be_nibbles(), bytes[1].to_be_nibbles()])
            .map(|[[_, first], _]| {
                std::convert::TryFrom::<u8>::try_from(0x0f & first).expect(NIBBLE_OVERFLOW)
            })
            .map(DelayTimerSrcTx::new)
            .parse(input)
    }
}

impl Default for DelayTimerSrcTx {
    fn default() -> Self {
        Self {
            dest: register::GpRegisters::V0,
        }
    }
}
