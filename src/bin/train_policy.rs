use anyhow::Result;
use std::env;
use std::fs::File;
use std::io::Read;
use super_duper_dragon::model::Position;

fn main() -> Result<()> {
    env::set_var("RUST_LOG", "info");
    env_logger::init();

    let mut f = File::open("./train_kifu_list.bin")?;
    let mut buf = vec![];
    f.read_to_end(&mut buf)?;
    let train_kifu: Vec<Position> = bincode::deserialize(&buf)?;
    log::info!("{}", train_kifu.len());
    Ok(())
}
