import database
import processing
import menus

# ==============================================================================
# -------------------------------- MAIN ENTRY ----------------------------------
# ==============================================================================
def main():
    # Load database registry and dnn detector models on boot
    registry = database.load_registry()
    net = processing.load_dnn_detector_if_available()

    while True:
        print("\n" + "="*40)
        print(f" MAIN MENU - ({len(registry['names'])} faces loaded)")
        print("="*40)
        print("1) Detect Faces (Fast Bounding Boxes Only)")
        print("2) Recognize Faces (Identifies People)")
        print("3) Add Known Faces (Register New People)")
        print("4) Manage Faces (Rename or Delete)")
        print("0) Quit")
        
        ch = input("\nSelect an option >>> ").strip()
        
        if ch == "0":
            print("Shutting down...")
            break
            
        if ch == "3":
            menus.register_new_faces(registry)
            continue
            
        if ch == "4":
            menus.manage_faces(registry)
            continue
            
        if ch not in ("1", "2"):
            print("Invalid option. Please try again.")
            continue

        # Run the smooth live session for detection or recognition
        menus.run_live_session(ch, registry, net)


if __name__ == "__main__":
    main()
