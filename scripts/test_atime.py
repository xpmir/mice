import os
import time
from pathlib import Path

def test_atime_behavior():
    test_file = Path("dummy_marker.done")

    print(f"--- Testing atime behavior on {test_file.absolute()} ---")

    # 1. Create a dummy file
    test_file.write_text("job finished")

    # 2. Wait a bit to ensure the clock ticks forward
    time.sleep(1.5)

    # Get initial atime
    initial_atime = test_file.stat().st_atime
    print(f"Initial atime:\t\t{initial_atime}")

    # 3. Simulate experimaestro constantly checking if the marker exists
    for _ in range(100):
        _ = test_file.is_file()  # This uses stat() under the hood
        _ = test_file.exists()

    # Wait another moment
    time.sleep(1.5)

    # Check atime after stat() calls
    stat_atime = test_file.stat().st_atime
    print(f"After is_file() checks:\t{stat_atime}")

    if initial_atime == stat_atime:
        print("✅ SUCCESS: is_file() did NOT update atime!")
    else:
        print("❌ WARNING: is_file() updated atime (unexpected!)")

    # 4. Now actually read the file contents
    _ = test_file.read_text()

    # Check atime after read()
    read_atime = test_file.stat().st_atime
    print(f"After read_text():\t{read_atime}")

    if read_atime > stat_atime:
        print("ℹ️ Note: Reading the file updated the atime (standard behavior without relatime).")
    else:
        print("ℹ️ Note: Reading didn't update atime (filesystem likely mounted with 'relatime').")

    # Cleanup
    test_file.unlink()

if __name__ == "__main__":
    test_atime_behavior()
