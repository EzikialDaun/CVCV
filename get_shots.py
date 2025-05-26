import os

import cv2


def check_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def get_facial_shots(path_input, path_output, alpha=0.2, step=1):
    check_directory(path_output)

    capture = cv2.VideoCapture(path_input)
    cnt_shot = 0

    # 최초 프레임
    first_ret, first_frame = capture.read()
    if first_ret:
        prev_hist = cv2.calcHist([first_frame], [0], None, [256], [0, 256])
        cv2.imwrite(f'{path_output}/{cnt_shot}_0.png', first_frame)
        print(f'{path_output}/{cnt_shot}_0.png')
        print()
        cnt_shot += 1
    else:
        return

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        if capture.get(cv2.CAP_PROP_POS_FRAMES) % step == 0:
            curr_hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            diff = cv2.compareHist(curr_hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
            print(f'diff: {diff}')
            if diff >= alpha:
                file_name = f'{path_output}/{cnt_shot}_{int(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)}.png'
                cv2.imwrite(file_name, frame)
                print(f"{file_name} saved.\n")
                cnt_shot += 1
            prev_hist = curr_hist

    # 종료
    if capture.isOpened():
        # 사용한 자원 해제
        capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    get_facial_shots(
        path_input='./data/Top.Gun.Maverick.2022.1080p.BluRay.x264.AAC5.1-[YTS.MX].mp4',
        path_output='./top_gun_maverick/indexed_shot',
        step=24, alpha=0.2)
