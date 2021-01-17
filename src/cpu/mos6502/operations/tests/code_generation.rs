use crate::cpu::mos6502::{
    microcode::*,
    operations::{address_mode, mnemonic, Instruction, MOps, Operation},
    Generate, MOS6502,
};

#[test]
fn should_generate_implied_address_mode_nop_machine_code() {
    let cpu = MOS6502::default();
    let op: Operation = Instruction::new(mnemonic::NOP, address_mode::Implied).into();
    let mc = op.generate(&cpu);

    // check Mops value is correct
    assert_eq!(MOps::new(1, 2, vec![]), mc);

    // validate mops -> vector looks correct
    assert_eq!(
        vec![vec![], vec![Microcode::IncPCRegister(IncPCRegister(1))]],
        Into::<Vec<Vec<Microcode>>>::into(mc)
    )
}

#[test]
fn should_generate_immediate_address_mode_lda_machine_code() {
    let cpu = MOS6502::default();
    let op: Operation = Instruction::new(mnemonic::LDA, address_mode::Immediate(0xff)).into();
    let mc = op.generate(&cpu);

    // check Mops value is correct
    assert_eq!(
        MOps::new(
            2,
            2,
            vec![Microcode::WriteAccRegister(WriteAccRegister(0xff))]
        ),
        mc
    );

    // validate mops -> vector looks correct
    assert_eq!(
        vec![
            vec![],
            vec![
                Microcode::WriteAccRegister(WriteAccRegister(0xff)),
                Microcode::IncPCRegister(IncPCRegister(2))
            ]
        ],
        Into::<Vec<Vec<Microcode>>>::into(mc)
    )
}
