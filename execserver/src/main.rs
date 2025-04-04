use axum::{routing::post, Router, Json};
use serde::Deserialize;
use std::{process::Output, time::Duration};
use tokio::io::AsyncWriteExt;

#[tokio::main]
async fn main() {
    let app = Router::new().route("/py_exec", post(py_exec));

    axum::Server::bind(&"0.0.0.0:8000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

#[derive(Deserialize)]
struct ExecRequest {
    code: String,
    timeout: u64,
    stdin: Option<String>,
}

async fn py_exec(Json(req): Json<ExecRequest>) -> String {
    let stdin = req.stdin.unwrap_or_default();
    run_python_code(&req.code, req.timeout, &stdin).await
}

async fn run_python_code(code: &str, timeout: u64, stdin: &str) -> String {
    let output = execute_with_timeout(code, stdin, Duration::from_secs(timeout)).await;
    format_output(output)
}

async fn execute_with_timeout(code: &str, stdin: &str, timeout: Duration) -> Option<Output> {
    let mut child = tokio::process::Command::new("python3")
        .arg("-c")
        .arg(code)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;

    if !stdin.is_empty() {
        let mut stdin_writer = child.stdin.take().unwrap();
        stdin_writer.write_all(stdin.as_bytes()).await.ok()?;
    }

    let output = tokio::time::timeout(timeout, child.wait_with_output()).await;
    output.ok()?
}

fn format_output(output: Option<Output>) -> String {
    match output {
        Some(out) if out.status.success() => format!("0\n{}", String::from_utf8_lossy(&out.stdout)),
        Some(out) => format!("1\n{}", String::from_utf8_lossy(&out.stderr)),
        None => "1\nTimeout".to_string(),
    }
}
