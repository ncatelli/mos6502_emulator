use crate::cpu::chip8::register;
use crate::cpu::chip8::{microcode::*, Chip8};
use crate::cpu::Generate;
use parcel::prelude::v1::*;

pub mod addressing_mode;

#[cfg(test)]
mod tests;

/// ToNibble provides methods for fetching the upper and lower nibble of a byte.
pub trait ToNibble {
    fn to_upper_nibble(&self) -> u8;
    fn to_lower_nibble(&self) -> u8;
}

impl ToNibble for u8 {
    fn to_upper_nibble(&self) -> u8 {
        (self & 0xf0) >> 4
    }

    fn to_lower_nibble(&self) -> u8 {
        self & 0x0f
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

pub fn matches_first_nibble_without_taking_input<'a>(
    opcode: u8,
) -> impl Parser<'a, &'a [(usize, u8)], u8> {
    move |input: &'a [(usize, u8)]| match input.get(0) {
        Some(&(pos, next)) if ((next & 0xf0) >> 4) == opcode => Ok(MatchStatus::Match {
            span: pos..pos + 1,
            remainder: &input[0..],
            inner: opcode,
        }),
        _ => Ok(MatchStatus::NoMatch(input)),
    }
}

/// Represents all valid opcodes for the CHIP-8 architecture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpcodeVariant {
    Cls(Cls),
    Ret(Ret),
    Jp(Jp<addressing_mode::Absolute>),
    Call(Call<addressing_mode::Absolute>),
    AddImmediate(Add<addressing_mode::Immediate>),
}

/// Provides a Parser type for the OpcodeVariant enum. Constructing an
/// OpcodeVariant from a stream of bytes.
pub struct OpcodeVariantParser;

impl<'a> Parser<'a, &'a [(usize, u8)], OpcodeVariant> for OpcodeVariantParser {
    fn parse(
        &self,
        input: &'a [(usize, u8)],
    ) -> parcel::ParseResult<&'a [(usize, u8)], OpcodeVariant> {
        parcel::one_of(vec![
            Cls::default().map(OpcodeVariant::Cls),
            Ret::default().map(OpcodeVariant::Ret),
            <Jp<addressing_mode::Absolute>>::default().map(OpcodeVariant::Jp),
            Call::default().map(OpcodeVariant::Call),
            <Add<addressing_mode::Immediate>>::default().map(OpcodeVariant::AddImmediate),
        ])
        .parse(input)
    }
}

impl Generate<Chip8, Vec<Microcode>> for OpcodeVariant {
    fn generate(self, cpu: &Chip8) -> Vec<Microcode> {
        match self {
            OpcodeVariant::Jp(op) => Generate::generate(op, cpu),
            OpcodeVariant::AddImmediate(op) => Generate::generate(op, cpu),
            // TODO: Empty placeholder representing a NOP
            _ => vec![],
        }
        .into_iter()
        .chain(vec![Microcode::Inc16bitRegister(
            // increment the PC by instruction size.
            Inc16bitRegister::new(register::WordRegisters::ProgramCounter, 2),
        )])
        .collect()
    }
}

/// Clear the display.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Cls {
    addressing_mode: addressing_mode::Implied,
}

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], Cls> for Cls {
    fn parse(&self, input: &'a [(usize, u8)]) -> parcel::ParseResult<&'a [(usize, u8)], Cls> {
        parcel::parsers::byte::expect_bytes(&[0x00, 0xe0])
            .map(|_| Cls::default())
            .parse(input)
    }
}

impl From<Cls> for u16 {
    fn from(_: Cls) -> Self {
        0x00e0
    }
}

/// Return from a subroutine.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Ret {
    addressing_mode: addressing_mode::Implied,
}

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], Ret> for Ret {
    fn parse(&self, input: &'a [(usize, u8)]) -> parcel::ParseResult<&'a [(usize, u8)], Ret> {
        parcel::parsers::byte::expect_bytes(&[0x00, 0xee])
            .map(|_| Ret::default())
            .parse(input)
    }
}

impl From<Ret> for u16 {
    fn from(_: Ret) -> Self {
        0x00ee
    }
}

/// Jp the associated value to the value of the specified register. Setting
/// the register to the sum.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Jp<A> {
    pub addressing_mode: A,
}

impl<A> Jp<A> {
    pub fn new(addressing_mode: A) -> Self {
        Self { addressing_mode }
    }
}

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], Jp<addressing_mode::Absolute>>
    for Jp<addressing_mode::Absolute>
{
    fn parse(
        &self,
        input: &'a [(usize, u8)],
    ) -> parcel::ParseResult<&'a [(usize, u8)], Jp<addressing_mode::Absolute>> {
        matches_first_nibble_without_taking_input(0x1)
            .and_then(|_| addressing_mode::Absolute::default())
            .map(Jp::new)
            .parse(input)
    }
}

impl From<Jp<addressing_mode::Absolute>> for OpcodeVariant {
    fn from(src: Jp<addressing_mode::Absolute>) -> Self {
        OpcodeVariant::Jp(src)
    }
}

impl Generate<Chip8, Vec<Microcode>> for Jp<addressing_mode::Absolute> {
    fn generate(self, _: &Chip8) -> Vec<Microcode> {
        vec![Microcode::Write16bitRegister(Write16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            u16::from(self.addressing_mode.addr()).wrapping_sub(2),
        ))]
    }
}

/// Call subroutine at nnn.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Call<A> {
    pub addressing_mode: A,
}

impl<A> Call<A> {
    pub fn new(addressing_mode: A) -> Self {
        Self { addressing_mode }
    }
}

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], Call<addressing_mode::Absolute>>
    for Call<addressing_mode::Absolute>
{
    fn parse(
        &self,
        input: &'a [(usize, u8)],
    ) -> parcel::ParseResult<&'a [(usize, u8)], Call<addressing_mode::Absolute>> {
        matches_first_nibble_without_taking_input(0x2)
            .and_then(|_| addressing_mode::Absolute::default())
            .map(Call::new)
            .parse(input)
    }
}

impl From<Call<addressing_mode::Absolute>> for OpcodeVariant {
    fn from(src: Call<addressing_mode::Absolute>) -> Self {
        OpcodeVariant::Call(src)
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

impl<'a> parcel::Parser<'a, &'a [(usize, u8)], Add<addressing_mode::Immediate>>
    for Add<addressing_mode::Immediate>
{
    fn parse(
        &self,
        input: &'a [(usize, u8)],
    ) -> parcel::ParseResult<&'a [(usize, u8)], Add<addressing_mode::Immediate>> {
        matches_first_nibble_without_taking_input(0x7)
            .and_then(|_| addressing_mode::Immediate::default())
            .map(Add::new)
            .parse(input)
    }
}

impl From<Add<addressing_mode::Immediate>> for OpcodeVariant {
    fn from(src: Add<addressing_mode::Immediate>) -> Self {
        OpcodeVariant::AddImmediate(src)
    }
}

impl Generate<Chip8, Vec<Microcode>> for Add<addressing_mode::Immediate> {
    fn generate(self, _: &Chip8) -> Vec<Microcode> {
        vec![Microcode::Inc8bitRegister(Inc8bitRegister::new(
            register::ByteRegisters::GpRegisters(self.addressing_mode.register),
            self.addressing_mode.value,
        ))]
    }
}
