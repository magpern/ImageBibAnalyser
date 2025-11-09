"""
Benchmark Tool — measures performance of bib detection on a set of images.

Usage:
  python benchmark.py --input ./photos --output benchmark.json --limit 500
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. Memory statistics will be limited.", file=sys.stderr)

try:
    from bib_finder import (
        gather_images,
        build_bib_regex,
        process_all,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from bib_finder import (
        gather_images,
        build_bib_regex,
        process_all,
    )


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.
    
    Returns:
        Dict with memory statistics in MB
    """
    if not HAS_PSUTIL:
        return {"available": 0, "used": 0, "percent": 0}
    
    process = psutil.Process()
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()
    
    return {
        "process_rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size
        "process_vms_mb": mem_info.vms / (1024 * 1024),  # Virtual Memory Size
        "system_available_mb": system_mem.available / (1024 * 1024),
        "system_used_mb": system_mem.used / (1024 * 1024),
        "system_percent": system_mem.percent,
    }


def run_benchmark(
    input_dir: Path,
    output_path: Path,
    limit: int = 500,
    workers: int = 4,
    min_digits: int = 2,
    max_digits: int = 6,
    min_conf: int = 60,
) -> Dict:
    """Run benchmark on image set.
    
    Args:
        input_dir: Directory with images
        output_path: Path to save benchmark results
        limit: Maximum number of images to process
        workers: Number of parallel workers
        min_digits: Minimum bib digits
        max_digits: Maximum bib digits
        min_conf: Minimum confidence threshold
        
    Returns:
        Dict with benchmark results
    """
    print(f"Starting benchmark on {limit} images...")
    print(f"Input directory: {input_dir}")
    print(f"Workers: {workers}")
    
    # Gather images
    exts = (".jpg", ".jpeg", ".png")
    paths = gather_images(input_dir, exts)
    
    if not paths:
        raise ValueError(f"No images found in {input_dir}")
    
    if len(paths) > limit:
        paths = paths[:limit]
        print(f"Limited to first {limit} images (found {len(paths)} total)")
    else:
        print(f"Processing {len(paths)} images")
    
    # Build regex
    bib_regex = build_bib_regex(min_digits, max_digits)
    
    # Get initial memory
    mem_before = get_memory_usage()
    
    # Run benchmark
    start_time = time.time()
    results = process_all(
        inputs=paths,
        bib_regex=bib_regex,
        min_conf=min_conf,
        workers=workers,
        rotations=(0, 90, -90, 180),
        psm_values=(6, 7, 11),
        annotate_dir=None,  # Skip annotation for benchmark
    )
    end_time = time.time()
    
    # Get final memory
    mem_after = get_memory_usage()
    
    # Calculate statistics
    elapsed_time = end_time - start_time
    images_per_second = len(paths) / elapsed_time if elapsed_time > 0 else 0
    
    total_bibs = sum(len(bibs) for _, bibs, _ in results)
    images_with_bibs = sum(1 for _, bibs, _ in results if bibs)
    
    benchmark_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_directory": str(input_dir),
        "configuration": {
            "images_processed": len(paths),
            "workers": workers,
            "min_digits": min_digits,
            "max_digits": max_digits,
            "min_confidence": min_conf,
        },
        "performance": {
            "elapsed_time_seconds": round(elapsed_time, 2),
            "images_per_second": round(images_per_second, 2),
            "total_time_minutes": round(elapsed_time / 60, 2),
        },
        "memory": {
            "before_mb": mem_before,
            "after_mb": mem_after,
            "delta_mb": {
                "process_rss": round(mem_after.get("process_rss_mb", 0) - mem_before.get("process_rss_mb", 0), 2),
                "process_vms": round(mem_after.get("process_vms_mb", 0) - mem_before.get("process_vms_mb", 0), 2),
            },
        },
        "results": {
            "total_images": len(paths),
            "images_with_bibs": images_with_bibs,
            "images_without_bibs": len(paths) - images_with_bibs,
            "total_bibs_detected": total_bibs,
            "avg_bibs_per_image": round(total_bibs / len(paths), 2) if paths else 0,
            "detection_rate": round(images_with_bibs / len(paths) * 100, 2) if paths else 0,
        },
    }
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Images processed: {len(paths)}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Throughput: {images_per_second:.2f} images/second")
    print(f"\nDetection Results:")
    print(f"  Images with bibs: {images_with_bibs}")
    print(f"  Images without bibs: {len(paths) - images_with_bibs}")
    print(f"  Total bibs detected: {total_bibs}")
    print(f"  Average bibs per image: {total_bibs/len(paths):.2f}" if paths else "  Average bibs per image: 0.00")
    print(f"  Detection rate: {images_with_bibs/len(paths)*100:.2f}%" if paths else "  Detection rate: 0.00%")
    print(f"\nMemory Usage:")
    print(f"  Process RSS: {mem_after.get('process_rss_mb', 0):.2f} MB")
    print(f"  Process VMS: {mem_after.get('process_vms_mb', 0):.2f} MB")
    if mem_before.get('process_rss_mb', 0) > 0:
        print(f"  RSS Delta: {benchmark_results['memory']['delta_mb']['process_rss']:.2f} MB")
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)
    
    return benchmark_results


def main():
    ap = argparse.ArgumentParser(description="Benchmark bib detection performance")
    ap.add_argument("--input", required=True, type=Path, help="Input directory with images")
    ap.add_argument("--output", required=True, type=Path, help="Output JSON file for results")
    ap.add_argument("--limit", type=int, default=500, help="Maximum number of images to process")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    ap.add_argument("--min-digits", type=int, default=2, help="Minimum bib digits")
    ap.add_argument("--max-digits", type=int, default=6, help="Maximum bib digits")
    ap.add_argument("--min-conf", type=int, default=60, help="Minimum OCR confidence")
    
    args = ap.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    try:
        run_benchmark(
            input_dir=args.input,
            output_path=args.output,
            limit=args.limit,
            workers=args.workers,
            min_digits=args.min_digits,
            max_digits=args.max_digits,
            min_conf=args.min_conf,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

