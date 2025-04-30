import os
import cv2
import shutil
import subprocess

COLMAP_EXE = r"C:\\Users\\ADMIN\\Downloads\\colmap-x64-windows-cuda\\bin\\colmap.exe"

VIDEO_PATH = r"C:\Users\ADMIN\OneDrive\Изображения\IMG_8049.mp4"
WORKING_DIR = r"C:\\Users\\ADMIN\\OneDrive\\Desktop\\Vision"
INPUT_IMAGE_DIR = os.path.join(WORKING_DIR, "input")
OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
DISTORTED_DIR = os.path.join(OUTPUT_DIR, "distorted")
DISTORTED_DB_PATH = os.path.join(DISTORTED_DIR, "database.db")
DISTORTED_SPARSE_PATH = os.path.join(DISTORTED_DIR, "sparse", "0")
UNDISTORTED_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
UNDISTORTED_SPARSE_PATH = os.path.join(OUTPUT_DIR, "sparse", "0")

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def extract_frames(video_path, output_folder, frame_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count, saved_count = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % max(1, (fps // frame_rate)) == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Разбивка завершена! Сохранено {saved_count} кадров в {output_folder}")

def run_colmap_command(command_args, success_message):
    try:
        subprocess.run(command_args, check=True)
        print(success_message)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении Colmap: {e}")
        exit(1)
    except FileNotFoundError:
        print("Ошибка: Colmap не найден! Проверьте путь к colmap.exe.")
        exit(1)

def check_colmap_exists():
    if not os.path.exists(COLMAP_EXE):
        print(f"Ошибка: файл {COLMAP_EXE} не найден! Проверьте путь.")
        exit(1)

def feature_extraction(image_dir, database_path):
    run_colmap_command([
        COLMAP_EXE, "feature_extractor",
        "--image_path", image_dir,
        "--database_path", database_path
    ], "Feature extraction completed successfully!")

def feature_matching(database_path):
    run_colmap_command([
        COLMAP_EXE, "exhaustive_matcher",
        "--database_path", database_path
    ], "Feature matching completed successfully!")

def sparse_reconstruction(database_path, image_dir, output_path):
    run_colmap_command([
        COLMAP_EXE, "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", output_path
    ], "Sparse reconstruction completed successfully!")

def undistort_images(sparse_path, input_image_dir, output_base_dir):
    run_colmap_command([
        COLMAP_EXE, "image_undistorter",
        "--image_path", input_image_dir,
        "--input_path", sparse_path,
        "--output_path", output_base_dir,
        "--output_type", "COLMAP",
        "--max_image_size", "2000"
    ], "Images undistorted successfully!")
    
    # Создаем нужную структуру папок, если её нет
    os.makedirs(os.path.join(output_base_dir, "sparse", "0"), exist_ok=True)
    
    # Перемещаем файлы из sparse в sparse/0, если они оказались не в том месте
    sparse_dir = os.path.join(output_base_dir, "sparse")
    sparse_0_dir = os.path.join(sparse_dir, "0")
    
    # Проверяем и перемещаем важные файлы
    for file_name in ["cameras.bin", "images.bin", "points3D.bin"]:
        src_path = os.path.join(sparse_dir, file_name)
        dst_path = os.path.join(sparse_0_dir, file_name)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)
            print(f"Перемещен файл {file_name} в правильную директорию.")

def main():
    print("Очистка папок...")
    clear_folder(INPUT_IMAGE_DIR)
    clear_folder(OUTPUT_DIR)
    print("Разбивка видео на кадры...")
    extract_frames(VIDEO_PATH, INPUT_IMAGE_DIR, frame_rate=2)
    print("Проверка Colmap...")
    check_colmap_exists()

    # 1. Разметка с искажёнными изображениями
    print("Создание структуры для distorted...")
    os.makedirs(os.path.join(DISTORTED_DIR, "sparse", "0"), exist_ok=True)

    print("Извлечение признаков для distorted...")
    feature_extraction(INPUT_IMAGE_DIR, DISTORTED_DB_PATH)

    print("Поиск соответствий для distorted...")
    feature_matching(DISTORTED_DB_PATH)

    print("Разреженная реконструкция для distorted...")
    sparse_reconstruction(DISTORTED_DB_PATH, INPUT_IMAGE_DIR, os.path.join(DISTORTED_DIR, "sparse"))

    # 2. Исправление искажений
    print("Исправление искажений...")
    undistort_images(DISTORTED_SPARSE_PATH, INPUT_IMAGE_DIR, OUTPUT_DIR)

    print("Готово!")

if __name__ == "__main__":
    main()
