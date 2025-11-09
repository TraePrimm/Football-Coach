import os

# Test path on E: drive
test_path = "E:/data/Football"
test_file = os.path.join(test_path, "test.txt")

print("=" * 50)
print("TESTING E: DRIVE ACCESS")
print("=" * 50)

# Test 1: Check if E: drive exists
e_drive_exists = os.path.exists('E:\\')
print(f"\n1. Does E: drive exist? {e_drive_exists}")
if not e_drive_exists:
    print("   ✗ E: drive not found! Cannot continue tests.")
    exit()

# Test 2: Try to create the directory
print(f"\n2. Attempting to create directory: {test_path}")
try:
    os.makedirs(test_path, exist_ok=True)
    # Actually verify it was created
    if os.path.exists(test_path) and os.path.isdir(test_path):
        print(f"   ✓ Directory verified to exist!")
    else:
        print(f"   ✗ Directory creation claimed success but directory doesn't exist!")
except Exception as e:
    print(f"   ✗ Failed to create directory: {type(e).__name__}: {e}")
    exit()

# Test 3: Try to write a file
print(f"\n3. Attempting to write test file: {test_file}")
test_content = "This is a test file created at " + str(os.path.getmtime(test_path) if os.path.exists(test_path) else "unknown time")
try:
    with open(test_file, "w") as f:
        f.write(test_content)
    
    # Actually verify the file exists
    if os.path.exists(test_file) and os.path.isfile(test_file):
        file_size = os.path.getsize(test_file)
        print(f"   ✓ File verified to exist!")
        print(f"   File size: {file_size} bytes")
    else:
        print(f"   ✗ File write claimed success but file doesn't exist!")
except Exception as e:
    print(f"   ✗ Failed to write file: {type(e).__name__}: {e}")

# Test 4: Try to read the file back
print(f"\n4. Attempting to read the test file")
try:
    if not os.path.exists(test_file):
        print(f"   ✗ Cannot read - file doesn't exist!")
    else:
        with open(test_file, "r") as f:
            content = f.read()
        
        # Verify we actually read something
        if content:
            print(f"   ✓ File read successfully!")
            print(f"   Content length: {len(content)} characters")
            print(f"   First 50 chars: {content[:50]}")
        else:
            print(f"   ✗ File read but content is empty!")
except Exception as e:
    print(f"   ✗ Failed to read file: {type(e).__name__}: {e}")

# Test 5: List directory contents
print(f"\n5. Listing directory contents")
try:
    if os.path.exists(test_path):
        files = os.listdir(test_path)
        print(f"   ✓ Directory contains {len(files)} item(s):")
        for item in files:
            full_path = os.path.join(test_path, item)
            item_type = "DIR" if os.path.isdir(full_path) else "FILE"
            print(f"      [{item_type}] {item}")
    else:
        print(f"   ✗ Directory doesn't exist!")
except Exception as e:
    print(f"   ✗ Failed to list directory: {type(e).__name__}: {e}")

# Test 6: Try to delete the file
print(f"\n6. Attempting to clean up test file")
try:
    if os.path.exists(test_file):
        os.remove(test_file)
        # Verify it's actually gone
        if not os.path.exists(test_file):
            print(f"   ✓ Test file verified deleted!")
        else:
            print(f"   ✗ Delete claimed success but file still exists!")
    else:
        print(f"   ⚠ Test file doesn't exist (nothing to delete)")
except Exception as e:
    print(f"   ✗ Failed to delete file: {type(e).__name__}: {e}")

print("\n" + "=" * 50)
print("TEST COMPLETE")
print("=" * 50)