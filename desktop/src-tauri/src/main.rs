#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

struct EngineProcess(Mutex<Option<Child>>);

#[tauri::command]
fn get_engine_status() -> String {
    "MoE Stream Engine running on port 8000".to_string()
}

fn main() {
    tauri::Builder::default()
        .manage(EngineProcess(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![get_engine_status])
        .setup(|app| {
            // Automatically launch native moe_engine server daemon if not already listening
            let child = Command::new("../../build/moe_engine.exe")
                .args(["serve", "--port", "8000", "--web-dir", "../../web/dist"])
                .spawn()
                .ok();

            if let Some(c) = child {
                let state = app.state::<EngineProcess>();
                *state.0.lock().unwrap() = Some(c);
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

