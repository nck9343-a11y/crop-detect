"""
摄像头实测标定 —— 直接量 GSD，不依赖标称参数
================================================
为什么要实测：
    标称参数（传感器尺寸、焦距）在廉价模组上经常不准，而且标称值反映不了
    镜头畸变、实际有效分辨率（MTF 差的镜头，标称 5MP 也解析不出 5MP 的细节）。
    用已知尺寸的参照物直接测，一步到位。

原理：
    在距镜头 D 米处放一把尺子，拍一张，量出 L 厘米在画面里占 N 个像素。
        GSD(该距离) = L*10 / N   毫米/像素
    GSD 与距离成正比，故任意高度 H 的 GSD 为：
        GSD(H) = GSD(D) * H / D

用法：
    1. 列出可用摄像头
        python Camera_calib.py --list

    2. 拍一张标定图（把尺子放在镜头正前方 1 米处，尽量与镜头平面平行）
        python Camera_calib.py --shot --dist 1.0

    3. 在弹出的窗口里，沿尺子点两下（比如 0 cm 和 20 cm 两个刻度），
       然后输入这两点的实际间距
        python Camera_calib.py --measure calib.jpg --dist 1.0

    4. 得到 GSD 后，脚本会直接换算各飞行高度的分辨能力
"""

import argparse
import cv2
import numpy as np

SMALL_PX, LARGE_PX = 32, 96          # COCO 尺寸分档（按边长）
HEIGHTS = [5, 10, 20, 30, 50]        # 换算用的飞行高度（米）


def list_cameras(max_idx=8):
    print('扫描可用摄像头（索引 0 起）...')
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, fr = cap.read()
            if ok:
                h, w = fr.shape[:2]
                # 试探最大分辨率：先请求一个很大的值，驱动会回落到支持的最大值
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
                mw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                mh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f'  [{i}] 默认 {w}x{h}   最大 {mw}x{mh}')
                found.append(i)
            cap.release()
    if not found:
        print('  未找到摄像头。检查是否插好、是否被其他程序占用。')
    return found


def shoot(index, out, want_max=True):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f'无法打开摄像头 {index}')
    if want_max:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
    print('按 空格 拍照，按 q 退出。把尺子放在镜头正前方，与镜头平面平行。')
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        disp = fr.copy()
        h, w = fr.shape[:2]
        cv2.putText(disp, f'{w}x{h}  SPACE=shoot  q=quit', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('calib', disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            cv2.imwrite(out, fr)
            print(f'已保存 {out}  ({w}x{h})')
            break
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def measure(path, dist_m):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f'读不到 {path}')
    h, w = img.shape[:2]
    pts = []

    # 大图缩小显示，但坐标换算回原图，避免损失精度
    scale = min(1.0, 1200 / max(h, w))
    disp0 = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()

    def on_click(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x / scale, y / scale))
            cv2.circle(disp0, (x, y), 5, (0, 0, 255), -1)
            if len(pts) == 2:
                cv2.line(disp0,
                         (int(pts[0][0] * scale), int(pts[0][1] * scale)),
                         (int(pts[1][0] * scale), int(pts[1][1] * scale)),
                         (0, 0, 255), 2)
            cv2.imshow('measure', disp0)

    print('在尺子上点两个刻度（比如 0 cm 与 20 cm），然后按任意键继续。')
    cv2.imshow('measure', disp0)
    cv2.setMouseCallback('measure', on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(pts) < 2:
        raise SystemExit('没点满两个点。')

    npx = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
    real_cm = float(input('这两点的实际间距是多少厘米？ '))

    gsd_at_d = real_cm * 10 / npx        # mm/px
    print()
    print('=' * 70)
    print(f'  图像尺寸      {w} x {h}')
    print(f'  两点间距      {npx:.1f} px  =  {real_cm} cm')
    print(f'  标定距离      {dist_m} m')
    print(f'  GSD@{dist_m}m     {gsd_at_d:.4f} mm/px')
    print('=' * 70)
    print()
    print(f'{"高度(m)":>8}{"GSD(mm/px)":>13}{"达32px需":>12}{"达96px需":>12}{"单帧覆盖(m)":>16}')
    print('-' * 70)
    for H in HEIGHTS:
        g = gsd_at_d * H / dist_m
        fw, fh = g * w / 1000, g * h / 1000
        print(f'{H:>8}{g:>13.2f}{g * SMALL_PX / 10:>11.1f}cm'
              f'{g * LARGE_PX / 10:>11.1f}cm{f"{fw:.1f} x {fh:.1f}":>16}')
    print('-' * 70)
    print(f'  5 mm 病斑在 30 m 处占 {5 / (gsd_at_d * 30 / dist_m):.2f} px')
    print()
    print('  提示：以上是几何分辨率上限。廉价针孔镜头的实际解析力通常低于此值，')
    print('        真实可用尺寸还要再打折扣。若要更准，可拍分辨率测试卡。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true', help='列出可用摄像头')
    ap.add_argument('--shot', action='store_true', help='拍一张标定图')
    ap.add_argument('--index', type=int, default=0, help='摄像头索引')
    ap.add_argument('--out', default='calib.jpg', help='标定图保存路径')
    ap.add_argument('--measure', help='在已有图上测量')
    ap.add_argument('--dist', type=float, default=1.0, help='标定时镜头到尺子的距离（米）')
    a = ap.parse_args()

    if a.list:
        list_cameras()
    elif a.shot:
        shoot(a.index, a.out)
        print(f'\n接着运行：python Camera_calib.py --measure {a.out} --dist {a.dist}')
    elif a.measure:
        measure(a.measure, a.dist)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
