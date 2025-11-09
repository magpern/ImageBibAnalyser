"""
HTML Report Generator — creates an HTML report summarizing bib detection results.

Usage:
  python generate_report.py --db bibdb.json --output report.html --annotated-dir annotated/
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

try:
    from bib_storage import BibStorage
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from bib_storage import BibStorage


def generate_html_report(
    storage: BibStorage,
    output_path: Path,
    annotated_dir: Path | None = None,
) -> None:
    """Generate HTML report from bib storage database.
    
    Args:
        storage: BibStorage instance
        output_path: Path to output HTML file
        annotated_dir: Optional directory with annotated images
    """
    stats = storage.get_stats()
    all_urls = storage.get_all_urls()
    
    # Collect data for report
    entries = []
    for url in all_urls:
        entry = storage.get_entry(url)
        if entry:
            entries.append(entry)
    
    # Sort by number of bibs detected (descending)
    entries.sort(key=lambda e: len(e.get('bibs', [])), reverse=True)
    
    # Generate HTML
    html_parts = []
    
    # HTML header
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Race Bib Detection Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .stat-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        .search-box {
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 5px;
        }
        .search-box input {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
        }
        .image-card {
            background: white;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .image-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }
        .image-info {
            padding: 10px;
        }
        .image-url {
            font-size: 0.9em;
            color: #7f8c8d;
            word-break: break-all;
            margin-bottom: 10px;
        }
        .bibs {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        .bib-badge {
            background-color: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Race Bib Detection Report</h1>
        <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">""" + str(stats['total_images']) + """</div>
            <div class="stat-label">Total Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">""" + str(stats['unique_bibs']) + """</div>
            <div class="stat-label">Unique Bibs</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">""" + str(stats['total_detections']) + """</div>
            <div class="stat-label">Total Detections</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">""" + f"{stats['avg_bibs_per_image']:.2f}" + """</div>
            <div class="stat-label">Avg Bibs per Image</div>
        </div>
    </div>
    
    <div class="search-box">
        <input type="text" id="searchInput" placeholder="Search by bib number or URL..." onkeyup="filterImages()">
    </div>
    
    <div class="image-grid" id="imageGrid">
""")
    
    # Generate image cards
    for entry in entries:
        url = entry['image_url']
        bibs = entry.get('bibs', [])
        local_path = entry.get('local_path')
        confidences = entry.get('confidences', {})
        
        # Try to find annotated image
        img_src = None
        if annotated_dir and local_path:
            annotated_path = Path(annotated_dir) / (Path(local_path).stem + "_annotated" + Path(local_path).suffix)
            if annotated_path.exists():
                img_src = str(annotated_path)
            elif Path(local_path).exists():
                img_src = str(local_path)
        elif local_path and Path(local_path).exists():
            img_src = str(local_path)
        
        # Create image card
        html_parts.append("""        <div class="image-card" data-bibs="""" + " ".join(bibs) + """" data-url="""" + quote(url) + """">
""")
        
        if img_src:
            html_parts.append(f"""            <img src="{quote(img_src)}" alt="Image" loading="lazy">
""")
        else:
            html_parts.append("""            <div style="height: 200px; background-color: #ecf0f1; display: flex; align-items: center; justify-content: center; color: #7f8c8d;">
                No preview available
            </div>
""")
        
        html_parts.append("""            <div class="image-info">
                <div class="image-url">""" + html_escape(url[:80] + ("..." if len(url) > 80 else "")) + """</div>
                <div class="bibs">
""")
        
        for bib in bibs:
            conf = confidences.get(bib, 0.0)
            html_parts.append(f"""                    <span class="bib-badge" title="Confidence: {conf:.1f}%">{html_escape(bib)}</span>
""")
        
        html_parts.append("""                </div>
            </div>
        </div>
""")
    
    html_parts.append("""    </div>
    
    <script>
        function filterImages() {
            const input = document.getElementById('searchInput');
            const filter = input.value.toLowerCase();
            const cards = document.getElementsByClassName('image-card');
            
            let visibleCount = 0;
            for (let card of cards) {
                const bibs = card.getAttribute('data-bibs').toLowerCase();
                const url = card.getAttribute('data-url').toLowerCase();
                if (bibs.includes(filter) || url.includes(filter)) {
                    card.style.display = '';
                    visibleCount++;
                } else {
                    card.style.display = 'none';
                }
            }
            
            // Show no results message if needed
            let noResults = document.getElementById('noResults');
            if (visibleCount === 0 && filter) {
                if (!noResults) {
                    noResults = document.createElement('div');
                    noResults.id = 'noResults';
                    noResults.className = 'no-results';
                    noResults.textContent = 'No images found matching your search.';
                    document.getElementById('imageGrid').appendChild(noResults);
                }
            } else if (noResults) {
                noResults.remove();
            }
        }
    </script>
</body>
</html>""")
    
    # Write HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    
    print(f"Generated HTML report: {output_path}")
    print(f"  - {stats['total_images']} images")
    print(f"  - {stats['unique_bibs']} unique bibs")
    print(f"  - {stats['total_detections']} total detections")


def html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#x27;'))


def main():
    ap = argparse.ArgumentParser(description="Generate HTML report from bib storage database")
    ap.add_argument("--db", required=True, type=Path, help="Path to bib storage database (JSON file)")
    ap.add_argument("--output", required=True, type=Path, help="Output HTML file path")
    ap.add_argument("--annotated-dir", type=Path, default=None, help="Directory with annotated images")
    
    args = ap.parse_args()
    
    if not args.db.exists():
        print(f"Error: Database file not found: {args.db}", file=sys.stderr)
        sys.exit(1)
    
    storage = BibStorage(args.db)
    generate_html_report(storage, args.output, args.annotated_dir)


if __name__ == "__main__":
    main()

