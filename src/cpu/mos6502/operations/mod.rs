use crate::address_map::{page::Page, Addressable};
use crate::cpu::{
    mos6502::{microcode::Microcode, register::*, Generate, Mos6502, IRQ_VECTOR_HH, IRQ_VECTOR_LL},
    register::Register,
    Cyclable, Offset,
};
use isa_mos6502::{addressing_mode, mnemonic, Instruction, InstructionVariant};
use parcel::{ParseResult, Parser};
use std::fmt::Debug;
use std::num::Wrapping;

#[cfg(test)]
mod tests;

/// bit_is_set takes a u8 value and a u8 representing the bit place returning a
/// bool if the place is set. This defaults to false if it is out of range.
const fn bit_is_set(value: u8, place: u8) -> bool {
    if place > 7 {
        false
    } else {
        ((value >> place) & 1) == 1
    }
}

/// This Trait provides addition that that signifies the overflow of a twos complement number.
trait AddTwosComplement<Rhs = Self> {
    type Output;

    /// Adds the left and right hand sides, returning the value and the boolean
    /// representation of the formula
    /// (!LHSMSB & !RHSMSB & C) || (LHSMSB & RHSMSB & !C).
    fn twos_complement_add(self, rhs: Rhs, carry: bool) -> (Self::Output, bool);
}

/// This Trait provides twos-complement subtraction.
trait SubTwosComplement<Rhs = Self> {
    type Output;

    /// subtracts the left and right hand sides returning a cary using twos complement
    fn twos_complement_sub(self, rhs: Rhs, carry: bool) -> (Self::Output, bool);
}

/// This Trait provides a rotate-left operation using the carry bit as shift
/// in and the 7 bit as the shift out.
trait Rol<Rhs = Self> {
    type Output;

    /// Shifts all bits left, rotating in the carry bit from the right and the
    /// 7th bit of the lhs to the carry bit.
    fn rol(self, rhs: Rhs, carry: bool) -> Self::Output;
}

/// This Trait provides a rotate-right operation using the carry bit as shift
/// in and the 0 bit as the shift out.
trait Ror<Rhs = Self> {
    type Output;

    /// Shifts all bits right, rotating in the carry bit from the left and the
    /// 0th bit of the lhs to the carry bit.
    fn ror(self, rhs: Rhs, carry: bool) -> Self::Output;
}

/// Represents a response that will yield a result that might or might not
/// result in wrapping, overflow or negative values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Operand<T> {
    carry: bool,
    negative: bool,
    zero: bool,
    inner: T,
}

impl<T> Operand<T> {
    fn with_flags(inner: T, carry: bool, negative: bool, zero: bool) -> Self {
        Self {
            carry,
            negative,
            zero,
            inner,
        }
    }

    fn unwrap(self) -> T {
        self.inner
    }
}

impl<T> PartialEq<T> for Operand<T>
where
    T: PartialEq + Copy,
{
    fn eq(&self, rhs: &T) -> bool {
        let lhs = self.unwrap();
        lhs == *rhs
    }
}

impl Operand<u8> {
    pub fn new(inner: u8) -> Self {
        Self {
            carry: false,
            negative: ((inner >> 7) & 1) == 1, // most significant bit set
            zero: inner == 0,
            inner,
        }
    }
}

impl std::ops::Add for Operand<u8> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let (lhs, rhs) = (self.unwrap(), other.unwrap());
        let (sum, carry) = lhs.overflowing_add(rhs);
        let negative = ((sum >> 7) & 1) == 1; // most significant bit set
        let zero = sum == 0;

        Self::with_flags(sum, carry, negative, zero)
    }
}

impl AddTwosComplement for Operand<u8> {
    type Output = Self;

    #[allow(clippy::nonminimal_bool)]
    fn twos_complement_add(self, other: Self, carry: bool) -> (Self::Output, bool) {
        let sum = self + other;
        let (lhs, rhs) = (self.unwrap(), other.unwrap());
        let overflow = (!bit_is_set(lhs, 7) && !bit_is_set(rhs, 7) && carry)
            || (bit_is_set(lhs, 7) && bit_is_set(rhs, 7) && !carry);

        (sum, overflow)
    }
}

impl std::ops::Sub for Operand<u8> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let (lhs, rhs) = (self.unwrap(), other.unwrap());
        let (difference, carry) = lhs.overflowing_sub(rhs);
        let negative = ((difference >> 7) & 1) == 1; // most significant bit set
        let zero = difference == 0;

        Self::with_flags(difference, carry, negative, zero)
    }
}

impl SubTwosComplement for Operand<u8> {
    type Output = Self;

    fn twos_complement_sub(self, other: Self, carry: bool) -> (Self::Output, bool) {
        let carry_bit = carry as u8; // 1 if true 0 if false
        let rhs_ones_complement = Operand::new(255 - other.unwrap()) + Operand::new(carry_bit);
        self.twos_complement_add(rhs_ones_complement, carry)
    }
}

impl std::ops::BitAnd for Operand<u8> {
    type Output = Self;

    fn bitand(self, other: Self) -> Self::Output {
        let (lhs, rhs) = (self.unwrap(), other.unwrap());
        let value = lhs & rhs;
        Self::new(value)
    }
}

impl std::ops::BitOr for Operand<u8> {
    type Output = Self;

    fn bitor(self, other: Self) -> Self::Output {
        let (lhs, rhs) = (self.unwrap(), other.unwrap());
        let value = lhs | rhs;
        Self::new(value)
    }
}

impl std::ops::BitXor for Operand<u8> {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self::Output {
        let (lhs, rhs) = (self.unwrap(), other.unwrap());
        let value = lhs ^ rhs;
        Self::new(value)
    }
}

impl Rol for Operand<u8> {
    type Output = Self;

    fn rol(self, other: Self, carry: bool) -> Self::Output {
        let (lhs, rhs) = (self.unwrap(), other.unwrap());
        let carry_in = carry as u8; // bool translates to 1 or 0 representing 0th bit.
        let carry_out = bit_is_set(lhs, 7);
        let shifted = Operand::new(lhs << rhs | carry_in);

        Operand::with_flags(shifted.unwrap(), carry_out, shifted.negative, shifted.zero)
    }
}

impl Ror for Operand<u8> {
    type Output = Self;

    fn ror(self, other: Self, carry: bool) -> Self::Output {
        let (lhs, rhs) = (self.unwrap(), other.unwrap());
        let carry_in = (carry as u8) << 7;
        let carry_out = bit_is_set(lhs, 0);
        let shifted = Operand::new(lhs >> rhs | carry_in);

        Operand::with_flags(shifted.unwrap(), carry_out, shifted.negative, shifted.zero)
    }
}

impl std::ops::Shl for Operand<u8> {
    type Output = Self;

    fn shl(self, other: Self) -> Self::Output {
        let (lhs, rhs) = (self.unwrap(), other.unwrap());
        let carry = bit_is_set(lhs, 7); // if the msb is set, shift to carry
        let value = Operand::new(lhs << rhs);

        Operand::with_flags(value.unwrap(), carry, value.negative, value.zero)
    }
}

impl std::ops::Shr for Operand<u8> {
    type Output = Self;

    fn shr(self, other: Self) -> Self::Output {
        let (lhs, rhs) = (self.unwrap(), other.unwrap());
        let carry = bit_is_set(lhs, 0); // if the lsb is set, shift to carry
        let value = Operand::new(lhs >> rhs);

        Operand::with_flags(value.unwrap(), carry, value.negative, value.zero)
    }
}

// addressing mode Unpackers

/// Provides a wrapper around the operation of unpacking an addressing mode and
/// adding an indirect offset to it. This appropriately handles for overflow
/// and returns the address as a u16.
const fn add_index_to_address(addr: u16, index: u8) -> u16 {
    addr.overflowing_add(index as u16).0
}

/// Provides a wrapper around the operation of unpacking a zeropage addressing
/// mode and adding an indirect offset to it. This appropriately handles for
/// overflow and returns the address as a u16.
const fn add_index_to_zeropage_address(zeropage_addr: u8, index: u8) -> u16 {
    zeropage_addr.overflowing_add(index).0 as u16
}

/// Provides a wrapper around the common operation of dereferencing an indexed
/// indirect address. This is effectively taking the value at
/// (Operand + Index, addr at Operand + Index + 1).
fn dereference_indexed_indirect_address(cpu: &Mos6502, base_addr: u8, index: u8) -> u16 {
    u16::from_le_bytes([
        cpu.address_map
            .read(base_addr.overflowing_add(index).0 as u16),
        cpu.address_map
            .read(base_addr.overflowing_add(index + 1).0 as u16),
    ])
}

/// Provides a wrapper around the operation of dereferencing an indirect
/// address and then adding an index to that indirect address. This is
/// effectively the value at (Operand, Operand + 1) + Index.
fn dereference_indirect_indexed_address(cpu: &Mos6502, base_addr: u8, index: u8) -> u16 {
    u16::from_le_bytes([
        cpu.address_map.read(base_addr as u16),
        cpu.address_map.read(base_addr.overflowing_add(1).0 as u16),
    ]) + index as u16
}

/// Provides a wrapper around the common operation of dereferencing and address
/// mode and retrieving the value stored at the specified address from the
/// address map. This value is then returned in a wrapper Operand.
fn dereference_address_to_operand(cpu: &Mos6502, addr: u16, index: u8) -> Operand<u8> {
    Operand::new(
        cpu.address_map
            .read(add_index_to_address(addr as u16, index)),
    )
}

/// Provides a wrapper around generating a 16-bit address from the stack
/// pointer. This exists as a function solely to not lose intent in the type
/// conversion to u16.
const fn stack_pointer_from_byte_value(value: u8) -> u16 {
    u16::from_le_bytes([value, 0x01])
}

/// Operations functions as a concrete wrapper around a microcode operation with
/// metadata around sizing and cycles. This trait does NOT represent a cycle
/// but rather the microcode equivalent of a CPU instruction.
#[derive(Debug, Clone, PartialEq)]
pub struct Operations {
    offset: usize,
    cycles: usize,
    microcode: Vec<Microcode>,
}

impl Operations {
    pub fn new(offset: usize, cycles: usize, microcode: Vec<Microcode>) -> Self {
        Self {
            offset,
            cycles,
            microcode,
        }
    }
}

impl Cyclable for Operations {
    fn cycles(&self) -> usize {
        self.cycles
    }
}

impl Offset for Operations {
    fn offset(&self) -> usize {
        self.offset
    }
}

impl From<Operations> for Vec<Vec<Microcode>> {
    fn from(src: Operations) -> Self {
        let cycles = src.cycles();
        let offset = src.offset() as u16;
        let mut mcs = vec![Vec::<Microcode>::new(); cycles - 1];

        mcs.push(
            src.microcode
                .into_iter()
                .chain(
                    vec![gen_inc_16bit_register_microcode!(WordRegisters::Pc, offset)].into_iter(),
                )
                .collect(),
        );
        mcs
    }
}

/// Dispatch a generate method to each corresponding generic types generate method.
impl Generate<Mos6502> for InstructionVariant {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        match *self {
            InstructionVariant::AdcAbsolute(am) => {
                Instruction::new(mnemonic::Adc, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::AdcAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Adc, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::AdcAbsoluteIndexedWithY(am) => {
                Instruction::new(mnemonic::Adc, addressing_mode::AbsoluteIndexedWithY(am))
                    .generate(cpu)
            }
            InstructionVariant::AdcIndirectYIndexed(am) => {
                Instruction::new(mnemonic::Adc, addressing_mode::IndirectYIndexed(am)).generate(cpu)
            }
            InstructionVariant::AdcImmediate(am) => {
                Instruction::new(mnemonic::Adc, addressing_mode::Immediate(am)).generate(cpu)
            }
            InstructionVariant::AdcXIndexedIndirect(am) => {
                Instruction::new(mnemonic::Adc, addressing_mode::XIndexedIndirect(am)).generate(cpu)
            }
            InstructionVariant::AdcZeroPage(am) => {
                Instruction::new(mnemonic::Adc, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::AdcZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Adc, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::AndAbsolute(am) => {
                Instruction::new(mnemonic::And, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::AndAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::And, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::AndAbsoluteIndexedWithY(am) => {
                Instruction::new(mnemonic::And, addressing_mode::AbsoluteIndexedWithY(am))
                    .generate(cpu)
            }
            InstructionVariant::AndIndirectYIndexed(am) => {
                Instruction::new(mnemonic::And, addressing_mode::IndirectYIndexed(am)).generate(cpu)
            }
            InstructionVariant::AndImmediate(am) => {
                Instruction::new(mnemonic::And, addressing_mode::Immediate(am)).generate(cpu)
            }
            InstructionVariant::AndXIndexedIndirect(am) => {
                Instruction::new(mnemonic::And, addressing_mode::XIndexedIndirect(am)).generate(cpu)
            }
            InstructionVariant::AndZeroPage(am) => {
                Instruction::new(mnemonic::And, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::AndZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::And, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::AslAbsolute(am) => {
                Instruction::new(mnemonic::Asl, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::AslAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Asl, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::AslAccumulator => {
                Instruction::new(mnemonic::Asl, addressing_mode::Accumulator).generate(cpu)
            }
            InstructionVariant::AslZeroPage(am) => {
                Instruction::new(mnemonic::Asl, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::AslZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Asl, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::BccRelative(am) => {
                Instruction::new(mnemonic::Bcc, addressing_mode::Relative(am)).generate(cpu)
            }
            InstructionVariant::BcsRelative(am) => {
                Instruction::new(mnemonic::Bcs, addressing_mode::Relative(am)).generate(cpu)
            }
            InstructionVariant::BeqRelative(am) => {
                Instruction::new(mnemonic::Beq, addressing_mode::Relative(am)).generate(cpu)
            }
            InstructionVariant::BmiRelative(am) => {
                Instruction::new(mnemonic::Bmi, addressing_mode::Relative(am)).generate(cpu)
            }
            InstructionVariant::BitAbsolute(am) => {
                Instruction::new(mnemonic::Bit, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::BitZeroPage(am) => {
                Instruction::new(mnemonic::Bit, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::BneRelative(am) => {
                Instruction::new(mnemonic::Bne, addressing_mode::Relative(am)).generate(cpu)
            }
            InstructionVariant::BplRelative(am) => {
                Instruction::new(mnemonic::Bpl, addressing_mode::Relative(am)).generate(cpu)
            }
            InstructionVariant::BrkImplied => {
                Instruction::new(mnemonic::Brk, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::BvcRelative(am) => {
                Instruction::new(mnemonic::Bvc, addressing_mode::Relative(am)).generate(cpu)
            }
            InstructionVariant::BvsRelative(am) => {
                Instruction::new(mnemonic::Bvs, addressing_mode::Relative(am)).generate(cpu)
            }
            InstructionVariant::ClcImplied => {
                Instruction::new(mnemonic::Clc, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::CldImplied => {
                Instruction::new(mnemonic::Cld, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::CliImplied => {
                Instruction::new(mnemonic::Cli, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::ClvImplied => {
                Instruction::new(mnemonic::Clv, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::CmpAbsolute(am) => {
                Instruction::new(mnemonic::Cmp, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::CmpAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Cmp, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::CmpAbsoluteIndexedWithY(am) => {
                Instruction::new(mnemonic::Cmp, addressing_mode::AbsoluteIndexedWithY(am))
                    .generate(cpu)
            }
            InstructionVariant::CmpIndirectYIndexed(am) => {
                Instruction::new(mnemonic::Cmp, addressing_mode::IndirectYIndexed(am)).generate(cpu)
            }
            InstructionVariant::CmpImmediate(am) => {
                Instruction::new(mnemonic::Cmp, addressing_mode::Immediate(am)).generate(cpu)
            }
            InstructionVariant::CmpXIndexedIndirect(am) => {
                Instruction::new(mnemonic::Cmp, addressing_mode::XIndexedIndirect(am)).generate(cpu)
            }
            InstructionVariant::CmpZeroPage(am) => {
                Instruction::new(mnemonic::Cmp, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::CmpZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Cmp, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::CpxAbsolute(am) => {
                Instruction::new(mnemonic::Cpx, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::CpxImmediate(am) => {
                Instruction::new(mnemonic::Cpx, addressing_mode::Immediate(am)).generate(cpu)
            }
            InstructionVariant::CpxZeroPage(am) => {
                Instruction::new(mnemonic::Cpx, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::CpyAbsolute(am) => {
                Instruction::new(mnemonic::Cpy, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::CpyImmediate(am) => {
                Instruction::new(mnemonic::Cpy, addressing_mode::Immediate(am)).generate(cpu)
            }
            InstructionVariant::CpyZeroPage(am) => {
                Instruction::new(mnemonic::Cpy, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::DecAbsolute(am) => {
                Instruction::new(mnemonic::Dec, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::DecAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Dec, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::DecZeroPage(am) => {
                Instruction::new(mnemonic::Dec, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::DecZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Dec, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::DexImplied => {
                Instruction::new(mnemonic::Dex, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::DeyImplied => {
                Instruction::new(mnemonic::Dey, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::EorAbsolute(am) => {
                Instruction::new(mnemonic::Eor, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::EorAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Eor, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::EorAbsoluteIndexedWithY(am) => {
                Instruction::new(mnemonic::Eor, addressing_mode::AbsoluteIndexedWithY(am))
                    .generate(cpu)
            }
            InstructionVariant::EorIndirectYIndexed(am) => {
                Instruction::new(mnemonic::Eor, addressing_mode::IndirectYIndexed(am)).generate(cpu)
            }
            InstructionVariant::EorImmediate(am) => {
                Instruction::new(mnemonic::Eor, addressing_mode::Immediate(am)).generate(cpu)
            }
            InstructionVariant::EorXIndexedIndirect(am) => {
                Instruction::new(mnemonic::Eor, addressing_mode::XIndexedIndirect(am)).generate(cpu)
            }
            InstructionVariant::EorZeroPage(am) => {
                Instruction::new(mnemonic::Eor, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::EorZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Eor, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::IncAbsolute(am) => {
                Instruction::new(mnemonic::Inc, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::IncAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Inc, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::IncZeroPage(am) => {
                Instruction::new(mnemonic::Inc, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::IncZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Inc, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::InxImplied => {
                Instruction::new(mnemonic::Inx, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::InyImplied => {
                Instruction::new(mnemonic::Iny, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::JmpAbsolute(am) => {
                Instruction::new(mnemonic::Jmp, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::JmpIndirect(am) => {
                Instruction::new(mnemonic::Jmp, addressing_mode::Indirect(am)).generate(cpu)
            }
            InstructionVariant::JsrAbsolute(am) => {
                Instruction::new(mnemonic::Jsr, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::LdaAbsolute(am) => {
                Instruction::new(mnemonic::Lda, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::LdaAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Lda, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::LdaAbsoluteIndexedWithY(am) => {
                Instruction::new(mnemonic::Lda, addressing_mode::AbsoluteIndexedWithY(am))
                    .generate(cpu)
            }
            InstructionVariant::LdaIndirectYIndexed(am) => {
                Instruction::new(mnemonic::Lda, addressing_mode::IndirectYIndexed(am)).generate(cpu)
            }
            InstructionVariant::LdaImmediate(am) => {
                Instruction::new(mnemonic::Lda, addressing_mode::Immediate(am)).generate(cpu)
            }
            InstructionVariant::LdaXIndexedIndirect(am) => {
                Instruction::new(mnemonic::Lda, addressing_mode::XIndexedIndirect(am)).generate(cpu)
            }
            InstructionVariant::LdaZeroPage(am) => {
                Instruction::new(mnemonic::Lda, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::LdaZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Lda, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::LdxAbsolute(am) => {
                Instruction::new(mnemonic::Ldx, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::LdxAbsoluteIndexedWithY(am) => {
                Instruction::new(mnemonic::Ldx, addressing_mode::AbsoluteIndexedWithY(am))
                    .generate(cpu)
            }
            InstructionVariant::LdxImmediate(am) => {
                Instruction::new(mnemonic::Ldx, addressing_mode::Immediate(am)).generate(cpu)
            }
            InstructionVariant::LdxZeroPage(am) => {
                Instruction::new(mnemonic::Ldx, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::LdxZeroPageIndexedWithY(am) => {
                Instruction::new(mnemonic::Ldx, addressing_mode::ZeroPageIndexedWithY(am))
                    .generate(cpu)
            }
            InstructionVariant::LdyAbsolute(am) => {
                Instruction::new(mnemonic::Ldy, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::LdyAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Ldy, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::LdyImmediate(am) => {
                Instruction::new(mnemonic::Ldy, addressing_mode::Immediate(am)).generate(cpu)
            }
            InstructionVariant::LdyZeroPage(am) => {
                Instruction::new(mnemonic::Ldy, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::LdyZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Ldy, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::LsrAbsolute(am) => {
                Instruction::new(mnemonic::Lsr, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::LsrAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Lsr, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::LsrAccumulator => {
                Instruction::new(mnemonic::Lsr, addressing_mode::Accumulator).generate(cpu)
            }
            InstructionVariant::LsrZeroPage(am) => {
                Instruction::new(mnemonic::Lsr, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::LsrZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Lsr, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::NopImplied => {
                Instruction::new(mnemonic::Nop, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::OraAbsolute(am) => {
                Instruction::new(mnemonic::Ora, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::OraAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Ora, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::OraAbsoluteIndexedWithY(am) => {
                Instruction::new(mnemonic::Ora, addressing_mode::AbsoluteIndexedWithY(am))
                    .generate(cpu)
            }
            InstructionVariant::OraIndirectYIndexed(am) => {
                Instruction::new(mnemonic::Ora, addressing_mode::IndirectYIndexed(am)).generate(cpu)
            }
            InstructionVariant::OraImmediate(am) => {
                Instruction::new(mnemonic::Ora, addressing_mode::Immediate(am)).generate(cpu)
            }
            InstructionVariant::OraXIndexedIndirect(am) => {
                Instruction::new(mnemonic::Ora, addressing_mode::XIndexedIndirect(am)).generate(cpu)
            }
            InstructionVariant::OraZeroPage(am) => {
                Instruction::new(mnemonic::Ora, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::OraZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Ora, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::PhaImplied => {
                Instruction::new(mnemonic::Pha, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::PhpImplied => {
                Instruction::new(mnemonic::Php, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::PlaImplied => {
                Instruction::new(mnemonic::Pla, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::PlpImplied => {
                Instruction::new(mnemonic::Plp, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::RolAbsolute(am) => {
                Instruction::new(mnemonic::Rol, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::RolAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Rol, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::RolAccumulator => {
                Instruction::new(mnemonic::Rol, addressing_mode::Accumulator).generate(cpu)
            }
            InstructionVariant::RolZeroPage(am) => {
                Instruction::new(mnemonic::Rol, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::RolZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Rol, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::RorAbsolute(am) => {
                Instruction::new(mnemonic::Ror, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::RorAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Ror, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::RorAccumulator => {
                Instruction::new(mnemonic::Ror, addressing_mode::Accumulator).generate(cpu)
            }
            InstructionVariant::RorZeroPage(am) => {
                Instruction::new(mnemonic::Ror, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::RorZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Ror, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::RtiImplied => {
                Instruction::new(mnemonic::Rti, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::RtsImplied => {
                Instruction::new(mnemonic::Rts, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::SbcAbsolute(am) => {
                Instruction::new(mnemonic::Sbc, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::SbcAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Sbc, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::SbcAbsoluteIndexedWithY(am) => {
                Instruction::new(mnemonic::Sbc, addressing_mode::AbsoluteIndexedWithY(am))
                    .generate(cpu)
            }
            InstructionVariant::SbcIndirectYIndexed(am) => {
                Instruction::new(mnemonic::Sbc, addressing_mode::IndirectYIndexed(am)).generate(cpu)
            }
            InstructionVariant::SbcImmediate(am) => {
                Instruction::new(mnemonic::Sbc, addressing_mode::Immediate(am)).generate(cpu)
            }
            InstructionVariant::SbcXIndexedIndirect(am) => {
                Instruction::new(mnemonic::Sbc, addressing_mode::XIndexedIndirect(am)).generate(cpu)
            }
            InstructionVariant::SbcZeroPage(am) => {
                Instruction::new(mnemonic::Sbc, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::SbcZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Sbc, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::StaAbsolute(am) => {
                Instruction::new(mnemonic::Sta, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::StaAbsoluteIndexedWithX(am) => {
                Instruction::new(mnemonic::Sta, addressing_mode::AbsoluteIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::StaAbsoluteIndexedWithY(am) => {
                Instruction::new(mnemonic::Sta, addressing_mode::AbsoluteIndexedWithY(am))
                    .generate(cpu)
            }
            InstructionVariant::StaIndirectYIndexed(am) => {
                Instruction::new(mnemonic::Sta, addressing_mode::IndirectYIndexed(am)).generate(cpu)
            }
            InstructionVariant::StaXIndexedIndirect(am) => {
                Instruction::new(mnemonic::Sta, addressing_mode::XIndexedIndirect(am)).generate(cpu)
            }
            InstructionVariant::StaZeroPage(am) => {
                Instruction::new(mnemonic::Sta, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::StaZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Sta, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::StxAbsolute(am) => {
                Instruction::new(mnemonic::Stx, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::StxZeroPage(am) => {
                Instruction::new(mnemonic::Stx, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::StxZeroPageIndexedWithY(am) => {
                Instruction::new(mnemonic::Stx, addressing_mode::ZeroPageIndexedWithY(am))
                    .generate(cpu)
            }
            InstructionVariant::StyAbsolute(am) => {
                Instruction::new(mnemonic::Sty, addressing_mode::Absolute(am)).generate(cpu)
            }
            InstructionVariant::StyZeroPage(am) => {
                Instruction::new(mnemonic::Sty, addressing_mode::ZeroPage(am)).generate(cpu)
            }
            InstructionVariant::StyZeroPageIndexedWithX(am) => {
                Instruction::new(mnemonic::Sty, addressing_mode::ZeroPageIndexedWithX(am))
                    .generate(cpu)
            }
            InstructionVariant::SecImplied => {
                Instruction::new(mnemonic::Sec, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::SedImplied => {
                Instruction::new(mnemonic::Sed, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::SeiImplied => {
                Instruction::new(mnemonic::Sei, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::TaxImplied => {
                Instruction::new(mnemonic::Tax, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::TayImplied => {
                Instruction::new(mnemonic::Tay, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::TsxImplied => {
                Instruction::new(mnemonic::Tsx, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::TxaImplied => {
                Instruction::new(mnemonic::Txa, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::TxsImplied => {
                Instruction::new(mnemonic::Txs, addressing_mode::Implied).generate(cpu)
            }
            InstructionVariant::TyaImplied => {
                Instruction::new(mnemonic::Tya, addressing_mode::Implied).generate(cpu)
            }
        }
    }
}

impl Cyclable for InstructionVariant {
    fn cycles(&self) -> usize {
        isa_mos6502::CycleCost::cycles(self)
    }
}

impl Offset for InstructionVariant {
    fn offset(&self) -> usize {
        isa_mos6502::ByteSized::byte_size(self)
    }
}

/// Macros to simplify definition of instruction set parsers. by hiding the
/// process of converting an instruction parser to its corresponding operation
macro_rules! inst_to_variant {
    ($inst:expr) => {
        $inst.map(Into::into)
    };
    ($mnemonic:expr, $addrmode:expr) => {
        Instruction::new($mnemonic, $addrmode).map(Into::into)
    };
}

/// Provides a wrapper type for parsing byte slices into an InstructionVariant.
pub struct VariantParser;

impl<'a> Parser<'a, &'a [u8], InstructionVariant> for VariantParser {
    fn parse(&self, input: &'a [u8]) -> ParseResult<&'a [u8], InstructionVariant> {
        let preparse_input = &input[0..];
        let byte_input: Vec<(usize, u8)> = input.iter().take(3).copied().map(|v| (0, v)).collect();

        match VariantParser.parse(&byte_input[..]) {
            Ok(parcel::MatchStatus::Match {
                span: _,
                remainder: _,
                inner,
            }) => {
                let byte_size = inner.offset();
                Ok(parcel::MatchStatus::Match {
                    span: 0..byte_size,
                    remainder: &preparse_input[0..byte_size],
                    inner,
                })
            }
            Ok(parcel::MatchStatus::NoMatch(_)) => {
                Ok(parcel::MatchStatus::NoMatch(&preparse_input))
            }

            Err(e) => Err(e),
        }
    }
}

impl<'a> Parser<'a, &'a [(usize, u8)], InstructionVariant> for VariantParser {
    fn parse(
        &self,
        input: &'a [(usize, u8)],
    ) -> ParseResult<&'a [(usize, u8)], InstructionVariant> {
        parcel::one_of(vec![
            inst_to_variant!(mnemonic::Adc, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Adc,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(
                mnemonic::Adc,
                addressing_mode::AbsoluteIndexedWithY::default()
            ),
            inst_to_variant!(mnemonic::Adc, addressing_mode::IndirectYIndexed::default()),
            inst_to_variant!(mnemonic::Adc, addressing_mode::Immediate::default()),
            inst_to_variant!(mnemonic::Adc, addressing_mode::XIndexedIndirect::default()),
            inst_to_variant!(mnemonic::Adc, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Adc,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::And, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::And,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(
                mnemonic::And,
                addressing_mode::AbsoluteIndexedWithY::default()
            ),
            inst_to_variant!(mnemonic::And, addressing_mode::IndirectYIndexed::default()),
            inst_to_variant!(mnemonic::And, addressing_mode::Immediate::default()),
            inst_to_variant!(mnemonic::And, addressing_mode::XIndexedIndirect::default()),
            inst_to_variant!(mnemonic::And, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::And,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Asl, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Asl,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Asl, addressing_mode::Accumulator),
            inst_to_variant!(mnemonic::Asl, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Asl,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Bcc, addressing_mode::Relative::default()),
            inst_to_variant!(mnemonic::Bcs, addressing_mode::Relative::default()),
            inst_to_variant!(mnemonic::Beq, addressing_mode::Relative::default()),
            inst_to_variant!(mnemonic::Bmi, addressing_mode::Relative::default()),
            inst_to_variant!(mnemonic::Bit, addressing_mode::Absolute::default()),
            inst_to_variant!(mnemonic::Bit, addressing_mode::ZeroPage::default()),
            inst_to_variant!(mnemonic::Bne, addressing_mode::Relative::default()),
            inst_to_variant!(mnemonic::Bpl, addressing_mode::Relative::default()),
            inst_to_variant!(mnemonic::Brk, addressing_mode::Implied::default()),
            inst_to_variant!(mnemonic::Bvc, addressing_mode::Relative::default()),
            inst_to_variant!(mnemonic::Bvs, addressing_mode::Relative::default()),
            inst_to_variant!(mnemonic::Clc, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Cld, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Cli, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Clv, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Cmp, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Cmp,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(
                mnemonic::Cmp,
                addressing_mode::AbsoluteIndexedWithY::default()
            ),
            inst_to_variant!(mnemonic::Cmp, addressing_mode::IndirectYIndexed::default()),
            inst_to_variant!(mnemonic::Cmp, addressing_mode::Immediate::default()),
            inst_to_variant!(mnemonic::Cmp, addressing_mode::XIndexedIndirect::default()),
            inst_to_variant!(mnemonic::Cmp, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Cmp,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Cpx, addressing_mode::Absolute::default()),
            inst_to_variant!(mnemonic::Cpx, addressing_mode::Immediate::default()),
            inst_to_variant!(mnemonic::Cpx, addressing_mode::ZeroPage::default()),
            inst_to_variant!(mnemonic::Cpy, addressing_mode::Absolute::default()),
            inst_to_variant!(mnemonic::Cpy, addressing_mode::Immediate::default()),
            inst_to_variant!(mnemonic::Cpy, addressing_mode::ZeroPage::default()),
            inst_to_variant!(mnemonic::Dec, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Dec,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Dec, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Dec,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Dex, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Dey, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Eor, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Eor,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(
                mnemonic::Eor,
                addressing_mode::AbsoluteIndexedWithY::default()
            ),
            inst_to_variant!(mnemonic::Eor, addressing_mode::IndirectYIndexed::default()),
            inst_to_variant!(mnemonic::Eor, addressing_mode::Immediate::default()),
            inst_to_variant!(mnemonic::Eor, addressing_mode::XIndexedIndirect::default()),
            inst_to_variant!(mnemonic::Eor, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Eor,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Inc, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Inc,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Inc, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Inc,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Inx, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Iny, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Jmp, addressing_mode::Absolute::default()),
            inst_to_variant!(mnemonic::Jmp, addressing_mode::Indirect::default()),
            inst_to_variant!(mnemonic::Jsr, addressing_mode::Absolute::default()),
            inst_to_variant!(mnemonic::Lda, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Lda,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(
                mnemonic::Lda,
                addressing_mode::AbsoluteIndexedWithY::default()
            ),
            inst_to_variant!(mnemonic::Lda, addressing_mode::IndirectYIndexed::default()),
            inst_to_variant!(mnemonic::Lda, addressing_mode::Immediate::default()),
            inst_to_variant!(mnemonic::Lda, addressing_mode::XIndexedIndirect::default()),
            inst_to_variant!(mnemonic::Lda, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Lda,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Ldx, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Ldx,
                addressing_mode::AbsoluteIndexedWithY::default()
            ),
            inst_to_variant!(mnemonic::Ldx, addressing_mode::Immediate::default()),
            inst_to_variant!(mnemonic::Ldx, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Ldx,
                addressing_mode::ZeroPageIndexedWithY::default()
            ),
            inst_to_variant!(mnemonic::Ldy, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Ldy,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Ldy, addressing_mode::Immediate::default()),
            inst_to_variant!(mnemonic::Ldy, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Ldy,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Lsr, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Lsr,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Lsr, addressing_mode::Accumulator),
            inst_to_variant!(mnemonic::Lsr, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Lsr,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Nop, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Ora, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Ora,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(
                mnemonic::Ora,
                addressing_mode::AbsoluteIndexedWithY::default()
            ),
            inst_to_variant!(mnemonic::Ora, addressing_mode::IndirectYIndexed::default()),
            inst_to_variant!(mnemonic::Ora, addressing_mode::Immediate::default()),
            inst_to_variant!(mnemonic::Ora, addressing_mode::XIndexedIndirect::default()),
            inst_to_variant!(mnemonic::Ora, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Ora,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Pha, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Php, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Pla, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Plp, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Rol, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Rol,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Rol, addressing_mode::Accumulator),
            inst_to_variant!(mnemonic::Rol, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Rol,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Ror, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Ror,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Ror, addressing_mode::Accumulator),
            inst_to_variant!(mnemonic::Ror, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Ror,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Rti, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Rts, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Sbc, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Sbc,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(
                mnemonic::Sbc,
                addressing_mode::AbsoluteIndexedWithY::default()
            ),
            inst_to_variant!(mnemonic::Sbc, addressing_mode::IndirectYIndexed::default()),
            inst_to_variant!(mnemonic::Sbc, addressing_mode::Immediate::default()),
            inst_to_variant!(mnemonic::Sbc, addressing_mode::XIndexedIndirect::default()),
            inst_to_variant!(mnemonic::Sbc, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Sbc,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Sta, addressing_mode::Absolute::default()),
            inst_to_variant!(
                mnemonic::Sta,
                addressing_mode::AbsoluteIndexedWithX::default()
            ),
            inst_to_variant!(
                mnemonic::Sta,
                addressing_mode::AbsoluteIndexedWithY::default()
            ),
            inst_to_variant!(mnemonic::Sta, addressing_mode::IndirectYIndexed::default()),
            inst_to_variant!(mnemonic::Sta, addressing_mode::XIndexedIndirect::default()),
            inst_to_variant!(mnemonic::Sta, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Sta,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Stx, addressing_mode::Absolute::default()),
            inst_to_variant!(mnemonic::Stx, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Stx,
                addressing_mode::ZeroPageIndexedWithY::default()
            ),
            inst_to_variant!(mnemonic::Sty, addressing_mode::Absolute::default()),
            inst_to_variant!(mnemonic::Sty, addressing_mode::ZeroPage::default()),
            inst_to_variant!(
                mnemonic::Sty,
                addressing_mode::ZeroPageIndexedWithX::default()
            ),
            inst_to_variant!(mnemonic::Sec, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Sed, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Sei, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Tax, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Tay, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Tsx, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Txa, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Txs, addressing_mode::Implied),
            inst_to_variant!(mnemonic::Tya, addressing_mode::Implied),
        ])
        .parse(input)
    }
}

impl<M, A> Offset for Instruction<M, A>
where
    M: Copy + Debug + PartialEq + isa_mos6502::ByteSized,
    A: Copy + Debug + PartialEq + isa_mos6502::ByteSized,
{
    fn offset(&self) -> usize {
        isa_mos6502::ByteSized::byte_size(&self.mnemonic)
            + isa_mos6502::ByteSized::byte_size(&self.addressing_mode)
    }
}

impl<M, A> Cyclable for Instruction<M, A>
where
    M: Copy + Debug + PartialEq + isa_mos6502::ByteSized,
    A: Copy + Debug + PartialEq + isa_mos6502::ByteSized,
    Self: isa_mos6502::CycleCost,
{
    fn cycles(&self) -> usize {
        isa_mos6502::CycleCost::cycles(self)
    }
}

// Arithmetic Operations

// Adc

impl Generate<Mos6502> for Instruction<mnemonic::Adc, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap(), 0);

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_add(rhs, cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Adc, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, cpu.x.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, indexed_addr, 0);

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_add(rhs, cpu.ps.carry);

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Adc, addressing_mode::AbsoluteIndexedWithY> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, cpu.y.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, indexed_addr, 0);

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_add(rhs, cpu.ps.carry);

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Adc, addressing_mode::IndirectYIndexed> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let zpage_base_addr = self.addressing_mode.unwrap();
        let indirect_addr =
            dereference_indirect_indexed_address(cpu, zpage_base_addr, cpu.y.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(cpu.address_map.read(indirect_addr));

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_add(rhs, cpu.ps.carry);

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(zpage_base_addr as u16).contains(indirect_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Adc, addressing_mode::Immediate> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(self.addressing_mode.unwrap());

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_add(rhs, cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Adc, addressing_mode::XIndexedIndirect> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let indirect_addr =
            dereference_indexed_indirect_address(cpu, self.addressing_mode.unwrap(), cpu.x.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(cpu.address_map.read(indirect_addr));

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_add(rhs, cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Adc, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), 0);
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, addr, 0);

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_add(rhs, cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Adc, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_zeropage_address(addr, cpu.x.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, indexed_addr, 0);

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_add(rhs, cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

// Sbc

impl Generate<Mos6502> for Instruction<mnemonic::Sbc, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap(), 0);

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_sub(rhs, cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sbc, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, cpu.x.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, indexed_addr, 0);

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_sub(rhs, cpu.ps.carry);

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sbc, addressing_mode::AbsoluteIndexedWithY> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, cpu.y.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, indexed_addr, 0);

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_sub(rhs, cpu.ps.carry);

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sbc, addressing_mode::IndirectYIndexed> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let zpage_base_addr = self.addressing_mode.unwrap();
        let indirect_addr =
            dereference_indirect_indexed_address(cpu, zpage_base_addr, cpu.y.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(cpu.address_map.read(indirect_addr));

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_sub(rhs, cpu.ps.carry);

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(zpage_base_addr as u16).contains(indirect_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sbc, addressing_mode::Immediate> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(self.addressing_mode.unwrap());

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_sub(rhs, cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sbc, addressing_mode::XIndexedIndirect> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let indirect_addr =
            dereference_indexed_indirect_address(cpu, self.addressing_mode.unwrap(), cpu.x.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(cpu.address_map.read(indirect_addr));

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_sub(rhs, cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sbc, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), 0);
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, addr, 0);

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_sub(rhs, cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sbc, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_zeropage_address(addr, cpu.x.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, indexed_addr, 0);

        // calculate overflow
        let (value, overflow) = lhs.twos_complement_sub(rhs, cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

// Bit-wise Operations

// And

impl Generate<Mos6502> for Instruction<mnemonic::And, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap(), 0);
        let value = lhs & rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::And, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, addr, index);
        let value = lhs & rhs;

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::And, addressing_mode::AbsoluteIndexedWithY> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.y.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, addr, index);
        let value = lhs & rhs;

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::And, addressing_mode::IndirectYIndexed> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let zpage_base_addr = self.addressing_mode.unwrap();
        let indirect_addr =
            dereference_indirect_indexed_address(cpu, zpage_base_addr, cpu.y.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(cpu.address_map.read(indirect_addr));
        let value = lhs & rhs;

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(zpage_base_addr as u16).contains(indirect_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::And, addressing_mode::Immediate> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(self.addressing_mode.unwrap());
        let value = lhs & rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::And, addressing_mode::XIndexedIndirect> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let indirect_addr =
            dereference_indexed_indirect_address(cpu, self.addressing_mode.unwrap(), cpu.x.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(cpu.address_map.read(indirect_addr));
        let value = lhs & rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::And, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap() as u16, 0);
        let value = lhs & rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::And, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let indexed_addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, indexed_addr, 0);
        let value = lhs & rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

// Asl

impl Generate<Mos6502> for Instruction<mnemonic::Asl, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let value = dereference_address_to_operand(cpu, addr, 0) << Operand::new(1u8);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Asl, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let value = dereference_address_to_operand(cpu, addr, index) << Operand::new(1u8);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(indexed_addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Asl, addressing_mode::Accumulator> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.acc.read()) << Operand::new(1u8);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Asl, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap() as u16;
        let value = dereference_address_to_operand(cpu, addr, 0) << Operand::new(1u8);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Asl, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let indexed_addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let value = dereference_address_to_operand(cpu, indexed_addr, 0) << Operand::new(1u8);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(indexed_addr, value.unwrap()),
            ],
        )
    }
}

// Bit

impl Generate<Mos6502> for Instruction<mnemonic::Bit, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap(), 0);
        let negative = bit_is_set(rhs.unwrap(), 7);
        let overflow = bit_is_set(rhs.unwrap(), 6);
        let value = lhs & rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Bit, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), 0);
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, addr, 0);
        let negative = bit_is_set(rhs.unwrap(), 7);
        let overflow = bit_is_set(rhs.unwrap(), 6);
        let value = lhs & rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Overflow, overflow),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
            ],
        )
    }
}

// Eor

impl Generate<Mos6502> for Instruction<mnemonic::Eor, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap(), 0);
        let value = lhs ^ rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Eor, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, addr, index);
        let value = lhs ^ rhs;

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Eor, addressing_mode::AbsoluteIndexedWithY> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.y.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, addr, index);
        let value = lhs ^ rhs;

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Eor, addressing_mode::IndirectYIndexed> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let zpage_base_addr = self.addressing_mode.unwrap();
        let indirect_addr =
            dereference_indirect_indexed_address(cpu, zpage_base_addr, cpu.y.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(cpu.address_map.read(indirect_addr));
        let value = lhs ^ rhs;

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(zpage_base_addr as u16).contains(indirect_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Eor, addressing_mode::Immediate> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(self.addressing_mode.unwrap());
        let value = lhs ^ rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Eor, addressing_mode::XIndexedIndirect> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let indirect_addr =
            dereference_indexed_indirect_address(cpu, self.addressing_mode.unwrap(), cpu.x.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(cpu.address_map.read(indirect_addr));
        let value = lhs ^ rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Eor, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap() as u16, 0);
        let value = lhs ^ rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Eor, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let indexed_addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, indexed_addr, 0);
        let value = lhs ^ rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

// Lsr

impl Generate<Mos6502> for Instruction<mnemonic::Lsr, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let value = dereference_address_to_operand(cpu, addr, 0) >> Operand::new(1u8);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Lsr, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let value = dereference_address_to_operand(cpu, addr, index) >> Operand::new(1u8);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(indexed_addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Lsr, addressing_mode::Accumulator> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.acc.read()) >> Operand::new(1u8);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Lsr, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap() as u16;
        let value = dereference_address_to_operand(cpu, addr, 0) >> Operand::new(1u8);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Lsr, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let indexed_addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let value = dereference_address_to_operand(cpu, indexed_addr, 0) >> Operand::new(1u8);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(indexed_addr, value.unwrap()),
            ],
        )
    }
}

// Ora

impl Generate<Mos6502> for Instruction<mnemonic::Ora, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap(), 0);
        let value = lhs | rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ora, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, addr, index);
        let value = lhs | rhs;

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ora, addressing_mode::AbsoluteIndexedWithY> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.y.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, addr, index);
        let value = lhs | rhs;

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ora, addressing_mode::IndirectYIndexed> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let zpage_base_addr = self.addressing_mode.unwrap();
        let indirect_addr =
            dereference_indirect_indexed_address(cpu, zpage_base_addr, cpu.y.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(cpu.address_map.read(indirect_addr));
        let value = lhs | rhs;

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(zpage_base_addr as u16).contains(indirect_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ora, addressing_mode::Immediate> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(self.addressing_mode.unwrap());
        let value = lhs | rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ora, addressing_mode::XIndexedIndirect> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let indirect_addr =
            dereference_indexed_indirect_address(cpu, self.addressing_mode.unwrap(), cpu.x.read());
        let lhs = Operand::new(cpu.acc.read());
        let rhs = Operand::new(cpu.address_map.read(indirect_addr));
        let value = lhs | rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ora, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap() as u16, 0);
        let value = lhs | rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ora, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let indexed_addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let lhs = Operand::new(cpu.acc.read());
        let rhs = dereference_address_to_operand(cpu, indexed_addr, 0);
        let value = lhs | rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

// Rol

impl Generate<Mos6502> for Instruction<mnemonic::Rol, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let lhs = dereference_address_to_operand(cpu, addr, 0);
        let value = lhs.rol(Operand::new(1u8), cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Rol, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let lhs = dereference_address_to_operand(cpu, addr, index);
        let value = lhs.rol(Operand::new(1u8), cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(indexed_addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Rol, addressing_mode::Accumulator> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.acc.read()).rol(Operand::new(1u8), cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Rol, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap() as u16;
        let lhs = dereference_address_to_operand(cpu, addr, 0);
        let value = lhs.rol(Operand::new(1u8), cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Rol, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let indexed_addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let lhs = dereference_address_to_operand(cpu, indexed_addr, 0);
        let value = lhs.rol(Operand::new(1u8), cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(indexed_addr, value.unwrap()),
            ],
        )
    }
}

// Ror

impl Generate<Mos6502> for Instruction<mnemonic::Ror, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let value =
            dereference_address_to_operand(cpu, addr, 0).ror(Operand::new(1u8), cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ror, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let value =
            dereference_address_to_operand(cpu, addr, index).ror(Operand::new(1u8), cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(indexed_addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ror, addressing_mode::Accumulator> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.acc.read()).ror(Operand::new(1u8), cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ror, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap() as u16;
        let value =
            dereference_address_to_operand(cpu, addr, 0).ror(Operand::new(1u8), cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ror, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let indexed_addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let value = dereference_address_to_operand(cpu, indexed_addr, 0)
            .ror(Operand::new(1u8), cpu.ps.carry);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, value.carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(indexed_addr, value.unwrap()),
            ],
        )
    }
}

// Branching

fn branch_on_case(
    cond: bool,
    branch_offset: i8,
    inst_offset: usize,
    cycles: usize,
    cpu: &Mos6502,
) -> Operations {
    let jmp_on_eq = (Wrapping(cpu.pc.read()) + Wrapping(branch_offset as u16)).0;
    let mc = if cond {
        vec![gen_write_16bit_register_microcode!(
            WordRegisters::Pc,
            // handle for underflow
            jmp_on_eq
        )]
    } else {
        vec![gen_write_16bit_register_microcode!(
            WordRegisters::Pc,
            // handle for underflow
            cpu.pc.read().overflowing_add(inst_offset as u16).0
        )]
    };

    // if the branch is true and that branch crosses a page boundary pay a 1 cycle penalty.
    let branch_penalty = match (cond, Page::from(cpu.pc.read()).contains(jmp_on_eq)) {
        (true, false) => 2,
        (true, true) => 1,
        _ => 0,
    };

    Operations::new(0, cycles + branch_penalty, mc)
}

// Bcc

impl Generate<Mos6502> for Instruction<mnemonic::Bcc, addressing_mode::Relative> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let offset = self.addressing_mode.unwrap();

        branch_on_case(!cpu.ps.carry, offset, self.offset(), self.cycles(), cpu)
    }
}

// Bcs

impl Generate<Mos6502> for Instruction<mnemonic::Bcs, addressing_mode::Relative> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let offset = self.addressing_mode.unwrap();

        branch_on_case(cpu.ps.carry, offset, self.offset(), self.cycles(), cpu)
    }
}

// Beq

impl Generate<Mos6502> for Instruction<mnemonic::Beq, addressing_mode::Relative> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let offset = self.addressing_mode.unwrap();

        branch_on_case(cpu.ps.zero, offset, self.offset(), self.cycles(), cpu)
    }
}

// Bmi

impl Generate<Mos6502> for Instruction<mnemonic::Bmi, addressing_mode::Relative> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let offset = self.addressing_mode.unwrap();

        branch_on_case(cpu.ps.negative, offset, self.offset(), self.cycles(), cpu)
    }
}

// Bne

impl Generate<Mos6502> for Instruction<mnemonic::Bne, addressing_mode::Relative> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let offset = self.addressing_mode.unwrap();

        branch_on_case(!cpu.ps.zero, offset, self.offset(), self.cycles(), cpu)
    }
}

// Bpl

impl Generate<Mos6502> for Instruction<mnemonic::Bpl, addressing_mode::Relative> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let offset = self.addressing_mode.unwrap();

        branch_on_case(!cpu.ps.negative, offset, self.offset(), self.cycles(), cpu)
    }
}

// Bvc

impl Generate<Mos6502> for Instruction<mnemonic::Bvc, addressing_mode::Relative> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let offset = self.addressing_mode.unwrap();

        branch_on_case(!cpu.ps.overflow, offset, self.offset(), self.cycles(), cpu)
    }
}

// Bvs

impl Generate<Mos6502> for Instruction<mnemonic::Bvs, addressing_mode::Relative> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let offset = self.addressing_mode.unwrap();

        branch_on_case(cpu.ps.overflow, offset, self.offset(), self.cycles(), cpu)
    }
}

// Clc

impl Generate<Mos6502> for Instruction<mnemonic::Clc, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, _: &Mos6502) -> Operations {
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_flag_set_microcode!(ProgramStatusFlags::Carry, false)],
        )
    }
}

// Cld

impl Generate<Mos6502> for Instruction<mnemonic::Cld, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, _: &Mos6502) -> Operations {
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_flag_set_microcode!(ProgramStatusFlags::Decimal, false)],
        )
    }
}

// Cli

impl Generate<Mos6502> for Instruction<mnemonic::Cli, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, _: &Mos6502) -> Operations {
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_flag_set_microcode!(
                ProgramStatusFlags::Interrupt,
                false
            )],
        )
    }
}

// Clv

impl Generate<Mos6502> for Instruction<mnemonic::Clv, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, _: &Mos6502) -> Operations {
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_flag_set_microcode!(ProgramStatusFlags::Overflow, false)],
        )
    }
}

// Cmp

impl Generate<Mos6502> for Instruction<mnemonic::Cmp, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap(), 0);
        let lhs = Operand::new(cpu.acc.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Cmp, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let base_addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(base_addr, index);
        let rhs = dereference_address_to_operand(cpu, indexed_addr, 0);
        let lhs = Operand::new(cpu.acc.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(base_addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Cmp, addressing_mode::AbsoluteIndexedWithY> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.y.read();
        let base_addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(base_addr, index);
        let rhs = dereference_address_to_operand(cpu, indexed_addr, 0);
        let lhs = Operand::new(cpu.acc.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(base_addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Cmp, addressing_mode::IndirectYIndexed> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.y.read();
        let base_addr = self.addressing_mode.unwrap();
        let indirect_addr = dereference_indirect_indexed_address(cpu, base_addr, index);
        let rhs = dereference_address_to_operand(cpu, indirect_addr, 0);
        let lhs = Operand::new(cpu.acc.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(base_addr as u16).contains(indirect_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Cmp, addressing_mode::Immediate> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addressing_mode::Immediate(am_value) = self.addressing_mode;
        let rhs = Operand::new(am_value);
        let lhs = Operand::new(cpu.acc.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Cmp, addressing_mode::XIndexedIndirect> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let indirect_addr =
            dereference_indexed_indirect_address(cpu, self.addressing_mode.unwrap(), index);
        let rhs = dereference_address_to_operand(cpu, indirect_addr, 0);
        let lhs = Operand::new(cpu.acc.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Cmp, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap() as u16, 0);
        let lhs = Operand::new(cpu.acc.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Cmp, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let base_addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let rhs = dereference_address_to_operand(cpu, base_addr, 0);
        let lhs = Operand::new(cpu.acc.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

// Cpx

impl Generate<Mos6502> for Instruction<mnemonic::Cpx, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap(), 0);
        let lhs = Operand::new(cpu.x.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Cpx, addressing_mode::Immediate> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addressing_mode::Immediate(am_value) = self.addressing_mode;
        let rhs = Operand::new(am_value);
        let lhs = Operand::new(cpu.x.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Cpx, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap() as u16, 0);
        let lhs = Operand::new(cpu.x.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

// Cpy

impl Generate<Mos6502> for Instruction<mnemonic::Cpy, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap(), 0);
        let lhs = Operand::new(cpu.y.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Cpy, addressing_mode::Immediate> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addressing_mode::Immediate(am_value) = self.addressing_mode;
        let rhs = Operand::new(am_value);
        let lhs = Operand::new(cpu.y.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Cpy, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let rhs = dereference_address_to_operand(cpu, self.addressing_mode.unwrap() as u16, 0);
        let lhs = Operand::new(cpu.y.read());
        let carry = lhs >= rhs;
        let diff = lhs - rhs;

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Carry, carry),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, diff.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, diff.zero),
            ],
        )
    }
}

// Dec

impl Generate<Mos6502> for Instruction<mnemonic::Dec, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let value = dereference_address_to_operand(cpu, addr, 0) - Operand::new(1);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Dec, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let value = dereference_address_to_operand(cpu, indexed_addr, 0) - Operand::new(1);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(indexed_addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Dec, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap() as u16;
        let value = dereference_address_to_operand(cpu, addr, 0) - Operand::new(1);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Dec, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_zeropage_address(addr, index);
        let value = dereference_address_to_operand(cpu, indexed_addr, 0) - Operand::new(1);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(indexed_addr, value.unwrap()),
            ],
        )
    }
}

// Dex

impl Generate<Mos6502> for Instruction<mnemonic::Dex, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.x.read()) - Operand::new(1);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_dec_8bit_register_microcode!(ByteRegisters::X, 1),
            ],
        )
    }
}

// Dey

impl Generate<Mos6502> for Instruction<mnemonic::Dey, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.x.read()) - Operand::new(1);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_dec_8bit_register_microcode!(ByteRegisters::Y, 1),
            ],
        )
    }
}

// Inc

impl Generate<Mos6502> for Instruction<mnemonic::Inc, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let value = dereference_address_to_operand(cpu, addr, 0) + Operand::new(1);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Inc, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let value = dereference_address_to_operand(cpu, indexed_addr, 0) + Operand::new(1);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(indexed_addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Inc, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap() as u16;
        let value = dereference_address_to_operand(cpu, addr, 0) + Operand::new(1);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(addr, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Inc, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_zeropage_address(addr, index);
        let value = dereference_address_to_operand(cpu, indexed_addr, 0) + Operand::new(1);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_memory_microcode!(indexed_addr, value.unwrap()),
            ],
        )
    }
}

// Inx

impl Generate<Mos6502> for Instruction<mnemonic::Inx, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.x.read()) + Operand::new(1);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::X, value.unwrap()),
            ],
        )
    }
}

// Iny

impl Generate<Mos6502> for Instruction<mnemonic::Iny, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.y.read()) + Operand::new(1);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Y, value.unwrap()),
            ],
        )
    }
}

// Jmp

impl Generate<Mos6502> for Instruction<mnemonic::Jmp, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, _: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();

        Operations::new(
            0,
            self.cycles(),
            vec![gen_write_16bit_register_microcode!(WordRegisters::Pc, addr)],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Jmp, addressing_mode::Indirect> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let indirect_addr = self.addressing_mode.unwrap();
        let lsb = cpu.address_map.read(indirect_addr);
        let msb = cpu.address_map.read(indirect_addr + 1);
        let addr = u16::from_le_bytes([lsb, msb]);

        Operations::new(
            0,
            self.cycles(),
            vec![gen_write_16bit_register_microcode!(WordRegisters::Pc, addr)],
        )
    }
}

// Jsr

impl Generate<Mos6502> for Instruction<mnemonic::Jsr, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();

        // grab the stack pointer and stack pointer - 1 for storing the PC
        let sph: u16 = stack_pointer_from_byte_value(cpu.sp.read());
        let spl: u16 = stack_pointer_from_byte_value(cpu.sp.read().wrapping_sub(1));

        // Add 2 to the program counter and grab as little-endian bytes.
        let [pcl, pch] = cpu.pc.read().wrapping_add(2).to_le_bytes();

        Operations::new(
            0,
            self.cycles(),
            vec![
                gen_write_memory_microcode!(sph, pch),
                gen_dec_8bit_register_microcode!(ByteRegisters::Sp, 1),
                gen_write_memory_microcode!(spl, pcl),
                gen_dec_8bit_register_microcode!(ByteRegisters::Sp, 1),
                gen_write_16bit_register_microcode!(WordRegisters::Pc, addr),
            ],
        )
    }
}

// Lda

impl Generate<Mos6502> for Instruction<mnemonic::Lda, addressing_mode::Immediate> {
    type Item = Operations;

    fn generate(&self, _: &Mos6502) -> Operations {
        let value = Operand::new(self.addressing_mode.unwrap());

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Lda, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = dereference_address_to_operand(cpu, self.addressing_mode.unwrap() as u16, 0);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Lda, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let value = dereference_address_to_operand(cpu, addr as u16, 0);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Lda, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addressing_mode::Absolute(addr) = self.addressing_mode;
        let value = Operand::new(cpu.address_map.read(addr));
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Lda, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let value = dereference_address_to_operand(cpu, addr, index);

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Lda, addressing_mode::AbsoluteIndexedWithY> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.y.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let value = dereference_address_to_operand(cpu, indexed_addr, 0);

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Lda, addressing_mode::IndirectYIndexed> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let zpage_base_addr = self.addressing_mode.unwrap();
        let indirect_addr =
            dereference_indirect_indexed_address(cpu, zpage_base_addr, cpu.y.read());
        let value = Operand::new(cpu.address_map.read(indirect_addr));

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(zpage_base_addr as u16).contains(indirect_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Lda, addressing_mode::XIndexedIndirect> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let indirect_addr =
            dereference_indexed_indirect_address(cpu, self.addressing_mode.unwrap(), cpu.x.read());
        let value = Operand::new(cpu.address_map.read(indirect_addr));

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

// Ldx

impl Generate<Mos6502> for Instruction<mnemonic::Ldx, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let value = dereference_address_to_operand(cpu, addr, 0);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::X, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ldx, addressing_mode::AbsoluteIndexedWithY> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.y.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let value = dereference_address_to_operand(cpu, indexed_addr, 0);

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::X, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ldx, addressing_mode::Immediate> {
    type Item = Operations;

    fn generate(&self, _: &Mos6502) -> Operations {
        let value = Operand::new(self.addressing_mode.unwrap());

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::X, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ldx, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = dereference_address_to_operand(cpu, self.addressing_mode.unwrap() as u16, 0);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::X, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ldx, addressing_mode::ZeroPageIndexedWithY> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.y.read();
        let addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let value = dereference_address_to_operand(cpu, addr as u16, 0);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::X, value.unwrap()),
            ],
        )
    }
}

// Ldy

impl Generate<Mos6502> for Instruction<mnemonic::Ldy, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let value = dereference_address_to_operand(cpu, addr, 0);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Y, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ldy, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = self.addressing_mode.unwrap();
        let indexed_addr = add_index_to_address(addr, index);
        let value = dereference_address_to_operand(cpu, indexed_addr, 0);

        // if the branch crosses a page boundary pay a 1 cycle penalty.
        let branch_penalty = if !Page::from(addr).contains(indexed_addr) {
            1
        } else {
            0
        };

        Operations::new(
            self.offset(),
            self.cycles() + branch_penalty,
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Y, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ldy, addressing_mode::Immediate> {
    type Item = Operations;

    fn generate(&self, _: &Mos6502) -> Operations {
        let value = Operand::new(self.addressing_mode.unwrap());

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Y, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ldy, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = dereference_address_to_operand(cpu, self.addressing_mode.unwrap() as u16, 0);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Y, value.unwrap()),
            ],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Ldy, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let value = dereference_address_to_operand(cpu, addr as u16, 0);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Y, value.unwrap()),
            ],
        )
    }
}

// Pha

impl Generate<Mos6502> for Instruction<mnemonic::Pha, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = cpu.acc.read();
        let sp = cpu.sp.read();

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_write_memory_microcode!(stack_pointer_from_byte_value(sp), value),
                gen_dec_8bit_register_microcode!(ByteRegisters::Sp, 1),
            ],
        )
    }
}

// Php

impl Generate<Mos6502> for Instruction<mnemonic::Php, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = cpu.ps.read();
        let sp = cpu.sp.read();

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_write_memory_microcode!(stack_pointer_from_byte_value(sp), value),
                gen_dec_8bit_register_microcode!(ByteRegisters::Sp, 1),
            ],
        )
    }
}

// Pla

impl Generate<Mos6502> for Instruction<mnemonic::Pla, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let sp = cpu.sp.read().overflowing_add(1).0;
        let value = dereference_address_to_operand(cpu, stack_pointer_from_byte_value(sp), 0);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_inc_8bit_register_microcode!(ByteRegisters::Sp, 1),
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

// Plp

impl Generate<Mos6502> for Instruction<mnemonic::Plp, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let sp = cpu.sp.read().overflowing_add(1).0;
        let value = dereference_address_to_operand(cpu, stack_pointer_from_byte_value(sp), 0);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_inc_8bit_register_microcode!(ByteRegisters::Sp, 1),
                gen_write_8bit_register_microcode!(ByteRegisters::Ps, value.unwrap()),
            ],
        )
    }
}

// Rti

impl Generate<Mos6502> for Instruction<mnemonic::Rti, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        // grab the program status
        let sp_psl: u16 = stack_pointer_from_byte_value(cpu.sp.read().wrapping_add(1));
        let sp = cpu.address_map.read(sp_psl);

        // grab the stack pointer and stack pointer - 1 for storing the PC
        let sp_pcl: u16 = stack_pointer_from_byte_value(cpu.sp.read().wrapping_add(2));
        let sp_pch: u16 = stack_pointer_from_byte_value(cpu.sp.read().wrapping_add(3));

        let (lsb, hsb) = (cpu.address_map.read(sp_pcl), cpu.address_map.read(sp_pch));
        let ret_addr = u16::from_le_bytes([lsb, hsb]);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_inc_8bit_register_microcode!(ByteRegisters::Sp, 1),
                gen_write_8bit_register_microcode!(ByteRegisters::Ps, sp),
                gen_inc_8bit_register_microcode!(ByteRegisters::Sp, 2),
                gen_write_16bit_register_microcode!(WordRegisters::Pc, ret_addr),
            ],
        )
    }
}

// Rts

impl Generate<Mos6502> for Instruction<mnemonic::Rts, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        // grab the stack pointer and stack pointer - 1 for storing the PC
        let spl: u16 = stack_pointer_from_byte_value(cpu.sp.read().wrapping_add(1));
        let sph: u16 = stack_pointer_from_byte_value(cpu.sp.read().wrapping_add(2));

        let (lsb, hsb) = (cpu.address_map.read(spl), cpu.address_map.read(sph));
        let ret_addr = u16::from_le_bytes([lsb, hsb]);

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_inc_8bit_register_microcode!(ByteRegisters::Sp, 2),
                gen_write_16bit_register_microcode!(WordRegisters::Pc, ret_addr),
            ],
        )
    }
}

// Sec

impl Generate<Mos6502> for Instruction<mnemonic::Sec, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, _: &Mos6502) -> Operations {
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_flag_set_microcode!(ProgramStatusFlags::Carry, true)],
        )
    }
}

// Sed

impl Generate<Mos6502> for Instruction<mnemonic::Sed, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, _: &Mos6502) -> Operations {
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_flag_set_microcode!(ProgramStatusFlags::Decimal, true)],
        )
    }
}

// Sei

impl Generate<Mos6502> for Instruction<mnemonic::Sei, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, _: &Mos6502) -> Operations {
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_flag_set_microcode!(ProgramStatusFlags::Interrupt, true)],
        )
    }
}

// Sta

impl Generate<Mos6502> for Instruction<mnemonic::Sta, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addressing_mode::Absolute(addr) = self.addressing_mode;
        let acc_val = cpu.acc.read();
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(addr, acc_val)],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sta, addressing_mode::AbsoluteIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let indexed_addr = add_index_to_address(self.addressing_mode.unwrap(), index);
        let acc_val = cpu.acc.read();
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(indexed_addr, acc_val)],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sta, addressing_mode::AbsoluteIndexedWithY> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.y.read();
        let indexed_addr = add_index_to_address(self.addressing_mode.unwrap(), index);
        let acc_val = cpu.acc.read();

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(indexed_addr, acc_val)],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sta, addressing_mode::IndirectYIndexed> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let indirect_addr =
            dereference_indirect_indexed_address(cpu, self.addressing_mode.unwrap(), cpu.y.read());
        let acc_val = cpu.acc.read();
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(indirect_addr, acc_val)],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sta, addressing_mode::XIndexedIndirect> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let indirect_addr =
            dereference_indexed_indirect_address(cpu, self.addressing_mode.unwrap(), cpu.x.read());
        let acc_val = cpu.acc.read();
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(indirect_addr, acc_val)],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sta, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap() as u16;
        let acc_val = cpu.acc.read();

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(addr, acc_val)],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sta, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let indexed_addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let acc_val = cpu.acc.read();

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(indexed_addr, acc_val)],
        )
    }
}

// Stx

impl Generate<Mos6502> for Instruction<mnemonic::Stx, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let value = cpu.x.read();
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(addr, value)],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Stx, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap() as u16;
        let value = cpu.x.read();

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(addr, value)],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Stx, addressing_mode::ZeroPageIndexedWithY> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.y.read();
        let indexed_addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let value = cpu.x.read();

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(indexed_addr, value)],
        )
    }
}

// Sty

impl Generate<Mos6502> for Instruction<mnemonic::Sty, addressing_mode::Absolute> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap();
        let value = cpu.y.read();
        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(addr, value)],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sty, addressing_mode::ZeroPage> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let addr = self.addressing_mode.unwrap() as u16;
        let value = cpu.y.read();

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(addr, value)],
        )
    }
}

impl Generate<Mos6502> for Instruction<mnemonic::Sty, addressing_mode::ZeroPageIndexedWithX> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let index = cpu.x.read();
        let indexed_addr = add_index_to_zeropage_address(self.addressing_mode.unwrap(), index);
        let value = cpu.y.read();

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_memory_microcode!(indexed_addr, value)],
        )
    }
}

// Tax

impl Generate<Mos6502> for Instruction<mnemonic::Tax, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.acc.read());

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::X, value.unwrap()),
            ],
        )
    }
}

// Tay

impl Generate<Mos6502> for Instruction<mnemonic::Tay, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.acc.read());

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Y, value.unwrap()),
            ],
        )
    }
}

// Tsx

impl Generate<Mos6502> for Instruction<mnemonic::Tsx, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.sp.read());

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::X, value.unwrap()),
            ],
        )
    }
}

// Txa

impl Generate<Mos6502> for Instruction<mnemonic::Txa, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.x.read());

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

// Tsx

impl Generate<Mos6502> for Instruction<mnemonic::Txs, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.x.read());

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![gen_write_8bit_register_microcode!(
                ByteRegisters::Sp,
                value.unwrap()
            )],
        )
    }
}

// Tya

impl Generate<Mos6502> for Instruction<mnemonic::Tya, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let value = Operand::new(cpu.y.read());

        Operations::new(
            self.offset(),
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Negative, value.negative),
                gen_flag_set_microcode!(ProgramStatusFlags::Zero, value.zero),
                gen_write_8bit_register_microcode!(ByteRegisters::Acc, value.unwrap()),
            ],
        )
    }
}

// Misc

// Brk

impl Generate<Mos6502> for Instruction<mnemonic::Brk, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, cpu: &Mos6502) -> Operations {
        let ps = cpu.ps.read();

        let sp_pcl: u16 = stack_pointer_from_byte_value(cpu.sp.read());
        let sp_pch: u16 = stack_pointer_from_byte_value(cpu.sp.read().wrapping_sub(1));
        let sp_ps: u16 = stack_pointer_from_byte_value(cpu.sp.read().wrapping_sub(2));

        // Add 1 to the program counter and grab as little-endian bytes.
        let [pcl, pch] = cpu.pc.read().wrapping_add(1).to_le_bytes();

        // Grab IRQ/Brk vector
        let irq_vector = u16::from_le_bytes([
            cpu.address_map.read(IRQ_VECTOR_LL),
            cpu.address_map.read(IRQ_VECTOR_HH),
        ]);

        Operations::new(
            0, // manually modified in the instruction
            self.cycles(),
            vec![
                gen_flag_set_microcode!(ProgramStatusFlags::Break, true),
                gen_flag_set_microcode!(ProgramStatusFlags::Interrupt, true),
                gen_write_memory_microcode!(sp_pcl, pcl),
                gen_dec_8bit_register_microcode!(ByteRegisters::Sp, 1),
                gen_write_memory_microcode!(sp_pch, pch),
                gen_dec_8bit_register_microcode!(ByteRegisters::Sp, 1),
                gen_write_memory_microcode!(sp_ps, ps), // PS Register
                gen_dec_8bit_register_microcode!(ByteRegisters::Sp, 1),
                gen_write_16bit_register_microcode!(WordRegisters::Pc, irq_vector),
            ],
        )
    }
}

// Nop

impl Generate<Mos6502> for Instruction<mnemonic::Nop, addressing_mode::Implied> {
    type Item = Operations;

    fn generate(&self, _: &Mos6502) -> Operations {
        Operations::new(self.offset(), self.cycles(), vec![])
    }
}
