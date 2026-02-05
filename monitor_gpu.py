#!/usr/bin/env python3
"""GPU Monitor - Real-time VRAM usage tracker

Usage:
    python monitor_gpu.py

This script monitors GPU memory usage in real-time, useful for:
- Verifying model distribution is working correctly
- Identifying memory leaks or bottlenecks
- Monitoring inference performance
"""
import time
import torch
from datetime import datetime

def clear_screen():
    """Clear terminal screen (cross-platform)."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def format_memory(bytes_val):
    """Format bytes to GB string."""
    return f"{bytes_val / (1024**3):.2f} GB"

def get_gpu_info():
    """Get GPU information and memory stats."""
    if not torch.cuda.is_available():
        return None
    
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        total = props.total_memory
        
        gpus.append({
            'id': i,
            'name': props.name,
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'allocated_pct': (allocated / total) * 100,
            'reserved_pct': (reserved / total) * 100,
        })
    
    return gpus

def draw_bar(percentage, width=40):
    """Draw a progress bar."""
    filled = int((percentage / 100) * width)
    bar = '█' * filled + '░' * (width - filled)
    return bar

def monitor_loop(interval=2.0):
    """Monitor GPU usage in a loop."""
    try:
        while True:
            clear_screen()
            
            print("=" * 100)
            print(f"GPU MEMORY MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 100)
            print()
            
            gpus = get_gpu_info()
            
            if gpus is None:
                print("❌ CUDA not available - no GPUs detected")
                print()
                print("Press Ctrl+C to exit")
                time.sleep(interval)
                continue
            
            for gpu in gpus:
                print(f"GPU {gpu['id']}: {gpu['name']}")
                print(f"  Total Memory:  {format_memory(gpu['total'])}")
                print()
                
                # Allocated memory
                print(f"  Allocated:     {format_memory(gpu['allocated'])} ({gpu['allocated_pct']:.1f}%)")
                print(f"  {draw_bar(gpu['allocated_pct'])}")
                print()
                
                # Reserved memory
                print(f"  Reserved:      {format_memory(gpu['reserved'])} ({gpu['reserved_pct']:.1f}%)")
                print(f"  {draw_bar(gpu['reserved_pct'])}")
                print()
                
                # Free memory
                free = gpu['total'] - gpu['reserved']
                free_pct = (free / gpu['total']) * 100
                print(f"  Free:          {format_memory(free)} ({free_pct:.1f}%)")
                print()
                print("-" * 100)
                print()
            
            # Expected distribution reminder
            print("Expected distribution (after models loaded):")
            print("  GPU 0: ~3-4 GB  (Summarizer - 4B model)")
            print("  GPU 1: ~16-18 GB (Buffer Agent - 27B model)")
            print()
            print("=" * 100)
            print(f"Refreshing every {interval}s... Press Ctrl+C to exit")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")

if __name__ == "__main__":
    print("Starting GPU monitor...")
    print("This will refresh every 2 seconds. Press Ctrl+C to stop.")
    print()
    time.sleep(1)
    monitor_loop()
