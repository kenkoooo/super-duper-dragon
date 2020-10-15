use anyhow::Result;
use shogiutil::{SfenBoard, SfenMove, UsiRequest, UsiResponse};
use std::io::{stdin, Read};

pub trait UsiPlayer {
    fn play(&mut self, request: UsiRequest) -> Vec<UsiResponse>;
    fn usi_play(&mut self) -> Result<()> {
        loop {
            let mut input = String::new();
            stdin().read_line(&mut input)?;
            log::info!("input: {}", input);

            let responses = self.play(UsiRequest::parse(input.trim())?);
            for response in responses {
                println!("{}", response.to_string());
            }
        }
    }
}
