"""
memory_guard.py — Lớp bảo vệ RAM cho Mac M1 16GB.

Dùng trước khi chạy Kronos training để đảm bảo Ollama
đã nhả RAM, tránh OOM crash.
"""
import time
import psutil
import requests


def unload_ollama() -> bool:
    """
    Yêu cầu Ollama giải phóng model khỏi RAM.
    Gọi trước khi chạy Kronos training.

    LƯU Ý QUAN TRỌNG:
    - ĐÚNG: POST /api/generate với keep_alive=0  → unload khỏi RAM, giữ file trên disk
    - SAI:  DELETE /api/delete                   → xóa model khỏi disk vĩnh viễn (mất 4.7GB)
    """
    try:
        # Gửi request unload — keep_alive: 0 báo Ollama nhả model ngay lập tức
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3:8b", "keep_alive": 0},
            timeout=15,
        )
    except requests.exceptions.ConnectionError:
        # Ollama không chạy → RAM đã trống, không cần làm gì thêm
        print("Ollama không chạy — RAM đã sẵn sàng.")
        return True
    except Exception as e:
        print(f"Cảnh báo unload_ollama: {e}")

    # Đợi RAM thực sự được giải phóng (tối đa 30 giây)
    print("Đang ép Ollama nhả RAM...")
    for elapsed in range(30):
        total_ollama_ram = 0.0
        for proc in psutil.process_iter(['name']):
            try:
                if 'ollama' in (proc.name() or '').lower():
                    total_ollama_ram += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process chết giữa chừng khi đang đọc — bỏ qua an toàn
                continue

        total_ollama_ram_gb = total_ollama_ram / (1024 ** 3)
        print(f"  [{elapsed+1}s] Ollama RAM: {total_ollama_ram_gb:.2f} GB")

        if total_ollama_ram_gb < 0.5:
            print("✓ Ollama đã nhả RAM thành công.")
            return True
        time.sleep(1)

    print("✗ Unload thất bại sau 30 giây — tiến hành kiểm tra thủ công.")
    return False


def available_ram_gb() -> float:
    """RAM khả dụng hiện tại (GB)."""
    return psutil.virtual_memory().available / (1024 ** 3)


def ram_is_safe_for_training(required_gb: float = 7.0) -> bool:
    """
    Barrier check trước khi training.
    Trả về False và in cảnh báo nếu không đủ RAM.
    """
    available = available_ram_gb()
    if available < required_gb:
        print(f"✗ RAM không đủ: {available:.1f} GB có sẵn < {required_gb} GB yêu cầu")
        print("  → Hủy training để tránh OOM. Thử lại sau khi đóng ứng dụng khác.")
        return False
    print(f"✓ RAM đủ: {available:.1f} GB có sẵn (yêu cầu {required_gb} GB)")
    return True


def prepare_for_training() -> bool:
    """
    Hàm tổng hợp: unload Ollama rồi kiểm tra RAM.
    Gọi 1 dòng này ở đầu kronos_trainer.py là đủ.

    Returns:
        True  → an toàn để training
        False → không đủ điều kiện, nên abort
    """
    print("=== Memory Guard: Chuẩn bị RAM cho Kronos Training ===")
    print(f"RAM trước khi dọn: {available_ram_gb():.1f} GB")

    unload_ollama()

    print(f"RAM sau khi dọn:   {available_ram_gb():.1f} GB")
    return ram_is_safe_for_training(required_gb=7.0)


if __name__ == "__main__":
    # Chạy thử độc lập để kiểm tra
    ok = prepare_for_training()
    if ok:
        print("\nSẵn sàng training Kronos!")
    else:
        print("\nChưa đủ điều kiện — kiểm tra lại RAM.")