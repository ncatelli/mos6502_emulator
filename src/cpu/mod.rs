pub trait Cyclable {
    fn cycles(&self) -> usize {
        1
    }
}

pub trait Offset {
    fn offset(&self) -> usize {
        1
    }
}

/// Generate fuctions to construct a representation of a set of operations to
/// transition a passed CPU state type, T, to a new state based on the calling
/// type.
pub trait Generate<T> {
    /// The type of the value output by the generate method.
    type Item;
    fn generate(&self, cpu: &T) -> Self::Item;
}

/// Defines a trait for implementing transformation on a CPU, returning the
/// modified CPU.
pub trait Execute<T> {
    fn execute(self, cpu: T) -> T;
}

/// Defines a trait for implementing transformation on a CPU, modifying a
/// references to a CPU in place. The difference between Execute and ExecuteMut
/// is that a ExecuteMut is implemented on the object (cpu) being modified.
pub trait ExecuteMut<T> {
    fn execute_mut(&mut self, operation: &T);
}

pub trait Cpu<T> {
    fn run(self, cycles: usize) -> StepState<T>;
}

/// Stores state between run invocations. The remaining field signifies noop cycles
/// between instructions. This is to function as a placeholder when a run
/// invocation returns while inbetween multi-cycle instructions.
pub enum StepState<T> {
    NotReady(usize, T),
    Ready(T),
}

impl<T> StepState<T> {
    pub fn new(cycle: usize, state: T) -> Self {
        match cycle {
            0 | 1 => Self::Ready(state),
            _ => Self::NotReady(cycle - 1, state),
        }
    }

    /// Decrements the cycle count left on StepState by 1.
    pub fn decrement(self) -> Self {
        match self {
            ss @ Self::Ready(_) => ss,
            Self::NotReady(1, state) => Self::Ready(state),
            Self::NotReady(0, state) => Self::Ready(state),
            Self::NotReady(cycles, state) => Self::NotReady(cycles - 1, state),
        }
    }

    pub fn unwrap(self) -> T {
        match self {
            Self::Ready(state) => state,
            Self::NotReady(_, state) => state,
        }
    }
}

impl<T> Cpu<T> for StepState<T>
where
    T: Cpu<T>,
{
    fn run(self, cycles: usize) -> StepState<T> {
        match self {
            StepState::Ready(cpu) => cpu.run(cycles),
            StepState::NotReady(remaining, cpu) if cycles < remaining => {
                StepState::NotReady(0, cpu)
            }
            StepState::NotReady(remaining, cpu) => cpu.run(cycles - remaining),
        }
    }
}

impl<T> From<T> for StepState<T> {
    fn from(src: T) -> Self {
        StepState::new(1, src)
    }
}

impl<T> From<StepState<T>> for (usize, T) {
    fn from(stepstate: StepState<T>) -> Self {
        match stepstate {
            StepState::Ready(state) => (0, state),
            StepState::NotReady(cycles, state) => (cycles, state),
        }
    }
}

#[macro_use]
pub mod mos6502;
pub mod chip8;
pub mod register;
