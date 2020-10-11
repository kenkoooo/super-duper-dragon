use serde::export::Formatter;
use std::iter::Enumerate;
use std::time::{Duration, Instant};

const DEFAULT_WRITE_SPAN_MILLIS: u128 = 1000;

pub struct ProgressBar<F, I> {
    enumerate: Enumerate<I>,
    start: Instant,
    prev_write: Instant,
    write: F,
    write_span: u128,
}

impl<F, I> ProgressBar<F, I>
where
    I: Iterator,
{
    pub fn new(iter: I, write: F) -> Self {
        Self {
            enumerate: iter.enumerate(),
            write,
            start: Instant::now(),
            prev_write: Instant::now(),
            write_span: DEFAULT_WRITE_SPAN_MILLIS,
        }
    }
}

impl<F, I, T> Iterator for ProgressBar<F, I>
where
    F: Fn(ProgressState),
    I: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((finished, next)) = self.enumerate.next() {
            if self.prev_write.elapsed().as_millis() >= self.write_span {
                let (size_hint_lower, size_hit_upper) = self.enumerate.size_hint();
                let remain = size_hit_upper.unwrap_or(size_hint_lower);
                let state = ProgressState {
                    elapsed: self.start.elapsed(),
                    finished,
                    remain,
                };
                (self.write)(state);
                self.prev_write = Instant::now();
            }

            Some(next)
        } else {
            None
        }
    }
}

pub trait ToProgressBar<F, I> {
    fn progress(self, f: F) -> ProgressBar<F, I>;
}
impl<F, I> ToProgressBar<F, I> for I
where
    I: Iterator,
{
    fn progress(self, f: F) -> ProgressBar<F, I> {
        ProgressBar::new(self, f)
    }
}

#[derive(Debug)]
pub struct ProgressState {
    pub remain: usize,
    pub finished: usize,
    pub elapsed: Duration,
}

impl ProgressState {
    pub fn estimated_remain_duration(&self) -> Option<Duration> {
        if self.finished > 0 {
            let remain =
                (self.elapsed.as_nanos() as u64) * (self.remain as u64) / (self.finished as u64);
            Some(Duration::from_nanos(remain))
        } else {
            None
        }
    }

    pub fn speed(&self) -> Option<Duration> {
        if self.finished > 0 {
            let speed = (self.elapsed.as_nanos() as u64) / (self.finished as u64);
            Some(Duration::from_nanos(speed))
        } else {
            None
        }
    }
}

impl std::fmt::Display for ProgressState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let total = self.remain + self.finished;
        write!(f, "{}/{}", self.finished, total)?;
        if total > 0 {
            let percent = (self.finished as f64) * 100.0 / (total as f64);
            write!(f, " {:.2}%", percent)?;
        }

        if let (Some(remain), Some(speed)) = (self.estimated_remain_duration(), self.speed()) {
            write!(
                f,
                " [{}<{}, {}]",
                format_duration(self.elapsed),
                format_duration(remain),
                format_speed(speed)
            )
        } else {
            write!(f, " [{}]", format_duration(self.elapsed))
        }
    }
}

fn format_speed(iter_duration: Duration) -> String {
    let second = Duration::from_secs(1);
    if second > iter_duration {
        let iter_per_sec = second.as_secs_f64() / iter_duration.as_secs_f64();
        format!("{:.2}it/s", iter_per_sec)
    } else {
        let sec_per_iter = iter_duration.as_secs_f64() / second.as_secs_f64();
        format!("{:.2}s/it", sec_per_iter)
    }
}

fn format_duration(elapsed: Duration) -> String {
    let total_sec = elapsed.as_secs();
    let hours = total_sec / 3600;
    let minutes = (total_sec % 3600) / 60;
    let seconds = total_sec % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}
