use anyhow::Result;
use rand::prelude::{SliceRandom, StdRng};
use rand::SeedableRng;
use std::fs::{read_dir, read_to_string};
use std::path::Path;

const PATH: &str = "./floodgate-kifu/wdoor2016/2016/";

struct KifuInfo<P> {
    path: P,
}

fn load_kifu_info<P: AsRef<Path>>(kifu_path: P) -> Option<KifuInfo<P>> {
    let rate_pattern = regex::Regex::new("^'(black|white)_rate:.*:(.*)").ok()?;
    let content = read_to_string(&kifu_path).ok()?;

    let mut black_rate = None;
    let mut white_rate = None;
    let mut is_resign = false;
    let mut moves = 0;

    for line in content.split("\n") {
        if let Some(caps) = rate_pattern.captures(line) {
            let side = &caps[1];
            let rate = caps[2].parse::<f64>().ok()?;
            if side == "black" {
                black_rate = Some(rate);
            } else {
                white_rate = Some(rate);
            }
        }

        if line.len() > 0 && (&line[..1] == "+" || &line[..1] == "-") {
            moves += 1
        }
        if line == "%TORYO" {
            is_resign = true;
        }
    }

    let black_rate = black_rate?;
    let white_rate = white_rate?;

    if !is_resign || moves <= 50 || min(black_rate, white_rate) < 2500.0 {
        None
    } else {
        Some(KifuInfo { path: kifu_path })
    }
}

fn min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

fn main() -> Result<()> {
    let mut kifu_infos = vec![];

    for entry in read_dir(PATH)? {
        let path = entry?.path();
        if path.is_file() {
            if let Some(info) = load_kifu_info(path) {
                kifu_infos.push(info);
            }
        }
    }

    let mut rng = StdRng::seed_from_u64(717);
    kifu_infos.shuffle(&mut rng);

    let train_count = kifu_infos.len() * 9 / 10;
    let test = kifu_infos.split_off(train_count);
    let train = kifu_infos;

    let mut train_files = String::new();
    for train_info in train {
        train_files += train_info.path.to_str().unwrap();
        train_files += "\n";
    }
    std::fs::write("train_kifu_list.txt", train_files)?;

    let mut test_files = String::new();
    for test_info in test {
        test_files += test_info.path.to_str().unwrap();
        test_files += "\n";
    }
    std::fs::write("test_kifu_list.txt", test_files)?;

    Ok(())
}
