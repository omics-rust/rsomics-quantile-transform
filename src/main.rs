mod cli;

fn main() -> std::process::ExitCode {
    use clap::Parser;
    cli::Cli::parse().run()
}
