use super::*;
use crate::cpu::chip8::{self, register, u12::u12, Chip8};
use crate::cpu::register::Register;

macro_rules! generate_parse_test {
    ($($testname:tt, $input_bytes:literal to $variant:expr,)*) => {
        $(
        #[test]
        fn $testname() {
            let input: Vec<(usize, u8)> = $input_bytes
                .to_be_bytes()
                .iter()
                .copied()
                .enumerate()
                .collect();
            assert_eq!(
                Ok(MatchStatus::Match {
                    span: 0..2,
                    remainder: &input[2..],
                    inner: $variant,
                }),
                OpcodeVariantParser.parse(&input[..])
            );
        }
        )*
    };
}

#[rustfmt::skip]
generate_parse_test!(
    should_parse_cls_opcode, 0x00e0u16 to Opcode::Cls,
    should_parse_ret_opcode, 0x00eeu16 to Opcode::Ret,
    should_parse_jump_absolute_opcode, 0x1fffu16 to Opcode::JpNonV0Indexed(u12::new(0xfff)),
    should_parse_jump_absolute_indexed_by_v0_opcode, 0xbfffu16 to Opcode::JpV0Indexed(u12::new(0xfff)),
    should_parse_load_absolute_into_i_opcode, 0xafffu16 to Opcode::LdAbsolute(u12::new(0xfff)),
    should_parse_load_immediate_into_i_opcode, 0x68ffu16 to Opcode::LdImmediate(GpRegisters::V8, 0xff),
    should_parse_load_byte_register_operation_opcode, 0x8010u16 to Opcode::LdVxVy(GpRegisters::V0, GpRegisters::V1),
    should_parse_load_byte_into_sound_timer_opcode, 0xf818u16 to Opcode::LdSoundTimerDestTx(GpRegisters::V8),
    should_parse_load_byte_into_delay_timer_opcode, 0xf815u16 to Opcode::LdDelayTimerDestTx(GpRegisters::V8),
    should_parse_load_byte_into_register_from_delay_timer_opcode, 0xf807u16 to Opcode::LdDelayTimerSrcTx(GpRegisters::V8),
    should_parse_load_bcd_from_vx_i_indirect_operation, 0xf833u16 to Opcode::LdBcd(GpRegisters::V8),
    should_parse_load_keypress_into_register_operation, 0xf80au16 to Opcode::LdK(GpRegisters::V8),
    should_parse_read_registers_from_memory_operation, 0xf265u16 to Opcode::ReadRegistersFromMemory(GpRegisters::V2),
    should_parse_store_registers_to_memory_operation, 0xf255u16 to Opcode::StoreRegistersToMemory(GpRegisters::V2),
    should_parse_call_opcode, 0x2fffu16 to Opcode::Call(u12::new(0xfff)),
    should_parse_add_immediate_opcode, 0x70ffu16 to Opcode::AddImmediate(GpRegisters::V0, 0xff),
    should_parse_add_i_register_indexed_opcode, 0xf01eu16 to Opcode::AddIRegisterIndexed(GpRegisters::V0),
    should_parse_add_vxvy_with_carry_operation, 0x8014u16 to Opcode::AddVxVy(GpRegisters::V0, GpRegisters::V1),
    should_parse_sub_vxvy_without_borrow_operation, 0x8015u16 to Opcode::Sub(GpRegisters::V0, GpRegisters::V1),
    should_parse_subn_vxvy_without_borrow_operation, 0x8017u16 to Opcode::Subn(GpRegisters::V0, GpRegisters::V1),
    should_parse_and_byte_register_operation_opcode, 0x8012u16 to Opcode::And(GpRegisters::V0, GpRegisters::V1),
    should_parse_or_byte_register_operation_opcode, 0x8011u16 to Opcode::Or(GpRegisters::V0, GpRegisters::V1),
    should_parse_xor_byte_register_operation_opcode, 0x8013u16 to Opcode::Xor(GpRegisters::V0, GpRegisters::V1),
    should_parse_shl_vxvy_with_flag_operation, 0x801eu16 to Opcode::Shl(GpRegisters::V0, GpRegisters::V1),
    should_parse_shr_vxvy_with_flag_operation, 0x8016u16 to Opcode::Shr(GpRegisters::V0, GpRegisters::V1),
    should_parse_se_immediate_operation_opcode, 0x30ffu16 to Opcode::SeImmediate(GpRegisters::V0, 0xff),
    should_parse_se_byte_register_operation_opcode, 0x5010u16 to Opcode::SeVxVy(GpRegisters::V0, GpRegisters::V1),
    should_parse_sne_immediate_operation_opcode, 0x40ffu16 to Opcode::SneImmediate(GpRegisters::V0, 0xff),
    should_parse_sne_byte_register_operation_opcode, 0x9010u16 to Opcode::SneVxVy(GpRegisters::V0, GpRegisters::V1),
    should_parse_rnd_immediate_operation_opcode, 0xc0ffu16 to Opcode::Rnd(GpRegisters::V0, 0xff),
    should_parse_skp_opcode, 0xe09eu16 to Opcode::Skp(GpRegisters::V0),
    should_parse_sknp_opcode, 0xe0a1u16 to Opcode::Sknp(GpRegisters::V0),
);

#[test]
fn should_generate_cls_instruction() {
    let cpu = Chip8::<()>::default();

    assert_eq!(
        vec![Microcode::SetDisplayRange(SetDisplayRange::new(
            (0, 0),
            (64, 32),
            false
        )),],
        Cls::default().generate(&cpu)
    );
}

#[test]
fn should_generate_ret_instruction() {
    let mut cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_pc_register(register::ProgramCounter::with_value(0x200));

    cpu.stack.write(0x0, 0x200);

    assert_eq!(
        vec![
            // save initial value
            Microcode::PopStack(PopStack::new(0x200)),
            // jump to absolute value - 2.
            Microcode::Write16bitRegister(Write16bitRegister::new(
                register::WordRegisters::ProgramCounter,
                0x1fe
            ))
        ],
        Ret.generate(&cpu)
    );
}

#[test]
fn should_generate_jump_absolute_with_pc_incrementer() {
    let cpu = Chip8::<()>::default().with_rng(|| 0u8);
    assert_eq!(
        vec![Microcode::Write16bitRegister(Write16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            0x1fe
        ))],
        Jp::<NonV0Indexed>::new(u12::new(0x200)).generate(&cpu)
    );
}

#[test]
fn should_generate_load_absolute_into_i_incrementer() {
    let cpu = Chip8::<()>::default().with_rng(|| 0u8);
    assert_eq!(
        vec![Microcode::Write16bitRegister(Write16bitRegister::new(
            register::WordRegisters::I,
            0xfff
        ))],
        Ld::new(addressing_mode::Absolute::new(u12::new(0xfff))).generate(&cpu)
    );
}

#[test]
fn should_generate_load_immediate_into_i_incrementer() {
    let cpu = Chip8::<()>::default().with_rng(|| 0u8);
    assert_eq!(
        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(register::GpRegisters::V8),
            0xff
        ))],
        Ld::new(addressing_mode::Immediate::new(
            register::GpRegisters::V8,
            0xff
        ))
        .generate(&cpu)
    );
}

#[test]
fn should_generate_load_byte_register_operation() {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0xff),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0x0f),
        );
    assert_eq!(
        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(register::GpRegisters::V0),
            0x0f
        ))],
        Ld::new(addressing_mode::VxVy::new(
            register::GpRegisters::V1,
            register::GpRegisters::V0
        ))
        .generate(&cpu)
    );
}

#[test]
fn should_generate_load_byte_into_sound_timer_operation() {
    let cpu = Chip8::<()>::default().with_rng(|| 0u8).with_gp_register(
        register::GpRegisters::V0,
        register::GeneralPurpose::<u8>::with_value(0xff),
    );
    assert_eq!(
        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::TimerRegisters(register::TimerRegisters::Sound),
            0xff
        ))],
        Ld::new(addressing_mode::SoundTimerDestTx::new(
            register::GpRegisters::V0,
        ))
        .generate(&cpu)
    );
}

#[test]
fn should_generate_load_byte_into_delay_timer_operation() {
    let cpu = Chip8::<()>::default().with_rng(|| 0u8).with_gp_register(
        register::GpRegisters::V0,
        register::GeneralPurpose::<u8>::with_value(0xff),
    );
    assert_eq!(
        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::TimerRegisters(register::TimerRegisters::Delay),
            0xff
        ))],
        Ld::new(addressing_mode::DelayTimerDestTx::new(
            register::GpRegisters::V0,
        ))
        .generate(&cpu)
    );
}

#[test]
fn should_generate_load_byte_into_register_from_delay_timer_operation() {
    let cpu = Chip8::<()>::default().with_rng(|| 0u8).with_timer_register(
        register::TimerRegisters::Delay,
        register::ClockDecrementing::with_value(0xff),
    );
    assert_eq!(
        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(register::GpRegisters::V0),
            0xff
        ))],
        Ld::new(addressing_mode::DelayTimerSrcTx::new(
            register::GpRegisters::V0,
        ))
        .generate(&cpu)
    );
}

#[test]
fn should_generate_load_bcd_from_vx_i_indirect_operation() {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0xfe),
        )
        .with_i_register(register::GeneralPurpose::<u16>::with_value(0x0100));

    assert_eq!(
        vec![
            Microcode::WriteMemory(WriteMemory::new(0x0100, 2)),
            Microcode::WriteMemory(WriteMemory::new(0x0101, 5)),
            Microcode::WriteMemory(WriteMemory::new(0x0102, 4)),
        ],
        LdBcd::new(addressing_mode::VxIIndirect::new(register::GpRegisters::V0,)).generate(&cpu)
    );
}

#[test]
fn should_generate_load_keypress_into_register_operation() {
    let cpu_with_input_interrupt = Chip8::<()>::default()
        .with_gp_register(
            register::GpRegisters::V8,
            register::GeneralPurpose::<u8>::with_value(0xab),
        )
        .with_interrupt(|| Some(chip8::Interrupt::KeyPress(chip8::KeyInputValue::Key4)));

    assert_eq!(
        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(GpRegisters::V8),
            0x04
        ))],
        LdK::new(register::GpRegisters::V8).generate(&cpu_with_input_interrupt)
    );

    let cpu_without_input_interrupt = Chip8::<()>::default()
        .with_gp_register(
            register::GpRegisters::V8,
            register::GeneralPurpose::<u8>::with_value(0xab),
        )
        .with_interrupt(|| None);

    assert_eq!(
        vec![Microcode::Dec16bitRegister(Dec16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            2,
        ))],
        LdK::new(register::GpRegisters::V8).generate(&cpu_without_input_interrupt)
    );
}

#[test]
fn should_generate_read_registers_from_memory_operation() {
    use crate::address_map::Addressable;

    let base_i_addr = 0x200;
    let mut cpu = Chip8::<()>::default()
        .with_i_register(register::GeneralPurpose::<u16>::with_value(base_i_addr));

    for v in 1..4 {
        let addr_offset = base_i_addr + (v as u16 - 1);
        cpu.address_space.write(addr_offset, v).unwrap();
    }

    assert_eq!(
        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V0),
                0x01
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V1),
                0x02
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V2),
                0x03
            )),
        ],
        ReadRegistersFromMemory::new(addressing_mode::VxIIndirect::new(register::GpRegisters::V2))
            .generate(&cpu)
    );
}

#[test]
fn should_generate_store_registers_to_memory_operation() {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0x01),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0x02),
        )
        .with_gp_register(
            register::GpRegisters::V2,
            register::GeneralPurpose::<u8>::with_value(0x03),
        )
        .with_i_register(register::GeneralPurpose::<u16>::with_value(0x0100));

    assert_eq!(
        vec![
            Microcode::WriteMemory(WriteMemory::new(0x0100, 0x01)),
            Microcode::WriteMemory(WriteMemory::new(0x0101, 0x02)),
            Microcode::WriteMemory(WriteMemory::new(0x0102, 0x03)),
        ],
        StoreRegistersToMemory::new(addressing_mode::VxIIndirect::new(register::GpRegisters::V2))
            .generate(&cpu)
    );
}

#[test]
fn should_generate_jump_absolute_indexed_by_v0_with_pc_incrementer() {
    let cpu = Chip8::<()>::default().with_rng(|| 0u8).with_gp_register(
        register::GpRegisters::V0,
        register::GeneralPurpose::with_value(0x05),
    );
    assert_eq!(
        vec![Microcode::Write16bitRegister(Write16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            0x203
        ))],
        Jp::<V0Indexed>::new(u12::new(0x200)).generate(&cpu)
    );
}

#[test]
fn should_generate_call_absolute_instruction() {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_pc_register(register::ProgramCounter::with_value(0x200));

    assert_eq!(
        vec![
            // save initial value
            Microcode::PushStack(PushStack::new(0x200)),
            // jump to absolute value - 2.
            Microcode::Write16bitRegister(Write16bitRegister::new(
                register::WordRegisters::ProgramCounter,
                0x3fe
            ))
        ],
        Call::new(u12::new(0x400)).generate(&cpu)
    );
}

#[test]
fn should_generate_add_immediate() {
    let cpu = Chip8::<()>::default().with_rng(|| 0u8);
    assert_eq!(
        vec![Microcode::Inc8bitRegister(Inc8bitRegister::new(
            register::ByteRegisters::GpRegisters(register::GpRegisters::V5),
            0xff
        ))],
        Add::new(addressing_mode::Immediate::new(
            register::GpRegisters::V5,
            0xff
        ))
        .generate(&cpu)
    );
}

#[test]
fn should_generate_add_i_register_indexed() {
    let cpu = Chip8::<()>::default().with_rng(|| 0u8).with_gp_register(
        register::GpRegisters::V5,
        register::GeneralPurpose::<u8>::with_value(0xff),
    );
    assert_eq!(
        vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
            register::WordRegisters::I,
            0xff
        ))],
        Add::new(addressing_mode::IRegisterIndexed::new(
            register::GpRegisters::V5
        ))
        .generate(&cpu)
    );
}

#[test]
fn should_generate_add_vxvy_with_carry_operation_that_overflows_when_sum_overflows_byte_capacity() {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0xfe),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0x05),
        );

    assert_eq!(
        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V0),
                0x03
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                0x01
            ))
        ],
        Add::new(addressing_mode::VxVy::new(GpRegisters::V1, GpRegisters::V0)).generate(&cpu)
    );
}

#[test]
fn should_generate_add_vxvy_with_carry_operation_that_does_not_set_overflow_when_under_byte() {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0xfe),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0x01),
        );

    assert_eq!(
        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V0),
                0xff
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                0x00
            ))
        ],
        Add::new(addressing_mode::VxVy::new(GpRegisters::V1, GpRegisters::V0)).generate(&cpu)
    );
}

#[test]
fn should_generate_set_borrow_for_sub_vxvy_if_vx_value_is_larger_than_vy_value() {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0xff),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0x05),
        );

    assert_eq!(
        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V0),
                0xfa
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                0x01
            ))
        ],
        Sub::new(addressing_mode::VxVy::new(GpRegisters::V1, GpRegisters::V0)).generate(&cpu)
    );
}

#[test]
fn should_generate_not_set_borrow_for_sub_vxvy_if_vx_value_is_larger_than_vy_value() {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0x01),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0xff),
        );

    assert_eq!(
        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V0),
                0x02
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                0x00
            ))
        ],
        Sub::new(addressing_mode::VxVy::new(GpRegisters::V1, GpRegisters::V0)).generate(&cpu)
    );
}

#[test]
fn should_generate_subn_vxvy_without_borrow_operation_that_doesnt_set_underflow_when_difference_underflows_byte_capacity(
) {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0xff),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0x05),
        );

    assert_eq!(
        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V0),
                0x06
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                0x00
            ))
        ],
        Subn::new(addressing_mode::VxVy::new(GpRegisters::V1, GpRegisters::V0)).generate(&cpu)
    );
}

#[test]
fn should_generate_subn_vxvy_without_underflow_operation_that_does_set_underflow_when_difference_fits_within_byte(
) {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0x01),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0xff),
        );

    assert_eq!(
        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V0),
                0xfe
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                0x01
            ))
        ],
        Subn::new(addressing_mode::VxVy::new(GpRegisters::V1, GpRegisters::V0)).generate(&cpu)
    );
}

#[test]
fn should_generate_and_byte_register_operation() {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0xff),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0x0f),
        );
    assert_eq!(
        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(register::GpRegisters::V0),
            0x0f
        ))],
        And::new(addressing_mode::VxVy::new(
            register::GpRegisters::V1,
            register::GpRegisters::V0
        ))
        .generate(&cpu)
    );
}

#[test]
fn should_generate_or_byte_register_operation() {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0xff),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0x0f),
        );
    assert_eq!(
        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(register::GpRegisters::V0),
            0xff
        ))],
        Or::new(addressing_mode::VxVy::new(
            register::GpRegisters::V1,
            register::GpRegisters::V0
        ))
        .generate(&cpu)
    );
}

#[test]
fn should_generate_shl_vxvy_with_flag_operation_that_overflows_byte_capacity() {
    let cpu = Chip8::<()>::default().with_gp_register(
        register::GpRegisters::V0,
        register::GeneralPurpose::<u8>::with_value(0x81u8),
    );

    assert_eq!(
        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                0x01
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V0),
                0x02
            ))
        ],
        Shl::new(addressing_mode::VxVy::new(GpRegisters::V1, GpRegisters::V0)).generate(&cpu)
    );
}

#[test]
fn should_generate_shl_vxvy_with_flag_operation_that_does_not_overflow_byte_capacity() {
    let cpu = Chip8::<()>::default().with_gp_register(
        register::GpRegisters::V0,
        register::GeneralPurpose::<u8>::with_value(0x01),
    );

    assert_eq!(
        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                0x0
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V0),
                0x2
            ))
        ],
        Shl::new(addressing_mode::VxVy::new(GpRegisters::V1, GpRegisters::V0)).generate(&cpu)
    );
}

#[test]
fn should_generate_shr_vxvy_with_flag_operation_that_overflows_byte_capacity() {
    let cpu = Chip8::<()>::default().with_gp_register(
        register::GpRegisters::V0,
        register::GeneralPurpose::<u8>::with_value(0x03u8),
    );

    assert_eq!(
        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                0x01
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V0),
                0x01
            ))
        ],
        Shr::new(addressing_mode::VxVy::new(GpRegisters::V1, GpRegisters::V0)).generate(&cpu)
    );
}

#[test]
fn should_generate_shr_vxvy_with_flag_operation_that_does_not_overflow_byte_capacity() {
    let cpu = Chip8::<()>::default().with_gp_register(
        register::GpRegisters::V0,
        register::GeneralPurpose::<u8>::with_value(0x8),
    );

    assert_eq!(
        vec![
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::Vf),
                0x0
            )),
            Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(GpRegisters::V0),
                0x4
            ))
        ],
        Shr::new(addressing_mode::VxVy::new(GpRegisters::V1, GpRegisters::V0)).generate(&cpu)
    );
}

#[test]
fn should_generate_xor_byte_register_operation() {
    let cpu = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0xff),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0x0f),
        );
    assert_eq!(
        vec![Microcode::Write8bitRegister(Write8bitRegister::new(
            register::ByteRegisters::GpRegisters(register::GpRegisters::V0),
            0xf0
        ))],
        Xor::new(addressing_mode::VxVy::new(
            register::GpRegisters::V1,
            register::GpRegisters::V0
        ))
        .generate(&cpu)
    );
}

#[test]
fn should_generate_se_immediate_operation() {
    let cpu_eq = Chip8::<()>::default().with_rng(|| 0u8).with_gp_register(
        register::GpRegisters::V0,
        register::GeneralPurpose::<u8>::with_value(0xff),
    );

    assert_eq!(
        vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            2
        ))],
        Se::new(addressing_mode::Immediate::new(
            register::GpRegisters::V0,
            0xff
        ))
        .generate(&cpu_eq)
    );

    let cpu_ne = Chip8::<()>::default().with_rng(|| 0u8).with_gp_register(
        register::GpRegisters::V0,
        register::GeneralPurpose::<u8>::with_value(0xff),
    );

    assert_eq!(
        Vec::<Microcode>::new(),
        Se::new(addressing_mode::Immediate::new(
            register::GpRegisters::V0,
            0x00
        ))
        .generate(&cpu_ne)
    );
}

#[test]
fn should_generate_se_byte_register_operation() {
    let cpu_eq = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0x0f),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0x0f),
        );
    assert_eq!(
        vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            2
        ))],
        Se::new(addressing_mode::VxVy::new(
            register::GpRegisters::V1,
            register::GpRegisters::V0
        ))
        .generate(&cpu_eq)
    );

    let cpu_ne = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0xff),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0x0f),
        );
    assert_eq!(
        Vec::<Microcode>::new(),
        Se::new(addressing_mode::VxVy::new(
            register::GpRegisters::V1,
            register::GpRegisters::V0
        ))
        .generate(&cpu_ne)
    );
}

#[test]
fn should_generate_sne_immediate_operation() {
    let cpu_eq = Chip8::<()>::default().with_rng(|| 0u8).with_gp_register(
        register::GpRegisters::V0,
        register::GeneralPurpose::<u8>::with_value(0xff),
    );

    assert_eq!(
        vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            2
        ))],
        Sne::new(addressing_mode::Immediate::new(
            register::GpRegisters::V0,
            0x00
        ))
        .generate(&cpu_eq)
    );

    let cpu_ne = Chip8::<()>::default().with_rng(|| 0u8).with_gp_register(
        register::GpRegisters::V0,
        register::GeneralPurpose::<u8>::with_value(0xff),
    );

    assert_eq!(
        Vec::<Microcode>::new(),
        Sne::new(addressing_mode::Immediate::new(
            register::GpRegisters::V0,
            0xff
        ))
        .generate(&cpu_ne)
    );
}

#[test]
fn should_generate_sne_byte_register_operation() {
    let cpu_eq = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0xff),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0x00),
        );
    assert_eq!(
        vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            2
        ))],
        Sne::new(addressing_mode::VxVy::new(
            register::GpRegisters::V1,
            register::GpRegisters::V0
        ))
        .generate(&cpu_eq)
    );

    let cpu_ne = Chip8::<()>::default()
        .with_rng(|| 0u8)
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0xff),
        )
        .with_gp_register(
            register::GpRegisters::V1,
            register::GeneralPurpose::<u8>::with_value(0xff),
        );
    assert_eq!(
        Vec::<Microcode>::new(),
        Sne::new(addressing_mode::VxVy::new(
            register::GpRegisters::V1,
            register::GpRegisters::V0
        ))
        .generate(&cpu_ne)
    );
}

#[test]
fn should_generate_rnd_immediate_operation() {
    let cpu = Chip8::<()>::default().with_rng(|| 0u8).with_gp_register(
        register::GpRegisters::V0,
        register::GeneralPurpose::<u8>::with_value(0x00),
    );

    // run many times
    for _ in 0..256 {
        // Assert 0x00 mask returns 0x00
        assert_eq!(
            vec![Microcode::Write8bitRegister(Write8bitRegister::new(
                register::ByteRegisters::GpRegisters(register::GpRegisters::V0),
                0
            ))],
            Rnd::new(addressing_mode::Immediate::new(
                register::GpRegisters::V0,
                0x00
            ))
            .generate(&cpu.clone())
        );

        // Assert 0x0f mask returns 0x00 - 0x0f
        let write_value = match Rnd::new(addressing_mode::Immediate::new(
            register::GpRegisters::V0,
            0x00,
        ))
        .generate(&cpu.clone())
        .get(0)
        {
            Some(Microcode::Write8bitRegister(op)) => op.value,
            _ => panic!("should container Write8BitRegister operation"),
        };

        assert!(write_value <= 0x0f);

        // Assert 0xf0 mask returns 0xf0 - 0xff
        let write_value = match Rnd::new(addressing_mode::Immediate::new(
            register::GpRegisters::V0,
            0xf0,
        ))
        .generate(&cpu.clone())
        .get(0)
        {
            Some(Microcode::Write8bitRegister(op)) => op.value,
            _ => panic!("should container Write8BitRegister operation"),
        };

        assert!(write_value <= 0xf0);
    }
}

#[test]
fn should_generate_skp_operation() {
    // a cpu with an input value set that matches.
    let cpu_some_eq = Chip8::<()>::default()
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0x0f),
        )
        .with_interrupt(|| Some(chip8::Interrupt::KeyPress(chip8::KeyInputValue::KeyF)));

    assert_eq!(
        vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            2
        ))],
        Skp::new(register::GpRegisters::V0).generate(&cpu_some_eq)
    );

    // a cpu with an input value that doesn't match.
    let cpu_some_ne = cpu_some_eq
        .clone()
        .with_interrupt(|| Some(chip8::Interrupt::KeyPress(chip8::KeyInputValue::Key0)));

    assert_eq!(
        Vec::<Microcode>::new(),
        Skp::new(register::GpRegisters::V0).generate(&cpu_some_ne)
    );

    // a cpu without a key pressed.
    let cpu_none = cpu_some_eq.clone().with_interrupt(|| None);

    assert_eq!(
        Vec::<Microcode>::new(),
        Skp::new(register::GpRegisters::V0).generate(&cpu_none)
    );
}

#[test]
fn should_generate_sknp_operation() {
    // a cpu with an input value set that matches.
    let cpu_some_eq = Chip8::<()>::default()
        .with_gp_register(
            register::GpRegisters::V0,
            register::GeneralPurpose::<u8>::with_value(0x0f),
        )
        .with_interrupt(|| Some(chip8::Interrupt::KeyPress(chip8::KeyInputValue::KeyF)));

    assert_eq!(
        Vec::<Microcode>::new(),
        Sknp::new(register::GpRegisters::V0).generate(&cpu_some_eq)
    );

    // a cpu with an input value that doesn't match.
    let cpu_some_ne = cpu_some_eq
        .clone()
        .with_interrupt(|| Some(chip8::Interrupt::KeyPress(chip8::KeyInputValue::Key0)));

    assert_eq!(
        vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            2
        ))],
        Sknp::new(register::GpRegisters::V0).generate(&cpu_some_ne)
    );

    // a cpu without a key pressed.
    let cpu_none = cpu_some_eq.clone().with_interrupt(|| None);

    assert_eq!(
        vec![Microcode::Inc16bitRegister(Inc16bitRegister::new(
            register::WordRegisters::ProgramCounter,
            2
        ))],
        Sknp::new(register::GpRegisters::V0).generate(&cpu_none)
    );
}
