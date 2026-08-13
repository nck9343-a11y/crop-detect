"""
无人机采集参数核算 —— 把 §5.1 的定性推导变成可验证的设计约束
================================================================
背景与动机：
    原报告 §5.1 由"小目标 AP 仅 0.161"推出必须降低 GSD、进而推出两级航线。
    但该 0.161 测自地面近摄图像，向航拍视角的迁移未经检验，
    因此那一步是跨域外推，不能作为结论。

    本脚本只使用成像光学关系，不依赖任何跨域假设：

        GSD = 像元尺寸 x 飞行高度 / 焦距

    以及 COCO 的尺寸分档定义（在图像上定义，与拍摄方式无关）：

        small   边长 <  32 px   (面积 <  1024 px^2)
        medium  边长 32-96 px
        large   边长 >  96 px

    由此可以严格回答一个问题：
        "要让直径 d 毫米的病斑在图像中达到 N 像素，飞行高度上限是多少？"
    这是纯光学结论，无论模型如何都成立，可直接作为采集方案的硬约束。

用法：
    python Gsd_planner.py

注意：
    下列相机参数为标称值，仅供估算。正式使用前必须以实际机型的
    传感器尺寸、有效像素与焦距核对，最好用已知尺寸标定物实测一次 GSD。
"""

import numpy as np

# ------------------------------------------------------------------ 相机参数
# name: (传感器宽 mm, 传感器高 mm, 图像宽 px, 图像高 px, 焦距 mm)
CAMERAS = {
    'DJI Mini 4 Pro (1/1.3", 48MP)':   (9.6, 7.2, 8064, 6048, 6.7),
    'DJI Air 3 广角 (1/1.3", 48MP)':    (9.6, 7.2, 8064, 6048, 6.7),
    'DJI Mavic 3 (4/3, 20MP)':         (17.3, 13.0, 5280, 3956, 12.29),
    'DJI Phantom 4 Pro (1", 20MP)':    (13.2, 8.8, 5472, 3648, 8.8),
    # 长焦镜头：焦距越长 GSD 越小，是唯一能在保持高度的同时提升分辨率的手段
    'Mavic 3 Pro 长焦 (1/1.3", 12MP)': (9.6, 7.2, 4032, 3024, 43.0),
}

# 关注的病斑物理尺寸（毫米）。依据报告 §5.1："单个病斑实际尺寸通常为数毫米量级"
LESION_MM = [3, 5, 10, 20, 50]

HEIGHTS = [1, 2, 3, 5, 10, 20, 30, 50, 100]      # 米
SMALL_PX, LARGE_PX = 32, 96                       # COCO 分档阈值（边长）

# 航线重叠率，正射拼接的常规取值
FWD_OVERLAP, SIDE_OVERLAP = 0.80, 0.70


def pixel_size_mm(sensor_mm, pixels):
    """单个像元的物理尺寸（毫米）。"""
    return sensor_mm / pixels


def gsd_mm(cam, height_m):
    """地面采样距离，毫米/像素。"""
    sw, _, pw, _, f = cam
    return pixel_size_mm(sw, pw) * (height_m * 1000.0) / f


def max_height_for(cam, lesion_mm, need_px):
    """使 lesion_mm 的目标达到 need_px 像素边长所允许的最大飞行高度（米）。"""
    sw, _, pw, _, f = cam
    gsd_need = lesion_mm / need_px          # mm/px
    return gsd_need * f / pixel_size_mm(sw, pw) / 1000.0


def footprint_m(cam, height_m):
    """单帧地面覆盖范围（米 x 米）。"""
    g = gsd_mm(cam, height_m) / 1000.0      # m/px
    _, _, pw, ph, _ = cam
    return g * pw, g * ph


def frames_per_ha(cam, height_m):
    """按给定重叠率覆盖 1 公顷所需的拍摄张数。"""
    w, h = footprint_m(cam, height_m)
    step_w = w * (1 - SIDE_OVERLAP)
    step_h = h * (1 - FWD_OVERLAP)
    return 10000.0 / (step_w * step_h)


def bucket(side_px):
    return 'small' if side_px < SMALL_PX else ('medium' if side_px < LARGE_PX else 'large')


def main():
    print('=' * 78)
    print('无人机采集参数核算')
    print('=' * 78)
    print('  依据: GSD = 像元尺寸 x 飞行高度 / 焦距   （纯光学关系）')
    print('  分档: COCO small <32px, medium 32-96px, large >96px（按边长）')
    print('  说明: 相机参数为标称值，正式使用前须以实机核对并实测标定')

    for name, cam in CAMERAS.items():
        sw, sh, pw, ph, f = cam
        ps_um = pixel_size_mm(sw, pw) * 1000
        print('\n' + '=' * 78)
        print(f'{name}')
        print(f'  传感器 {sw}x{sh} mm，{pw}x{ph} px，像元 {ps_um:.2f} um，焦距 {f} mm')
        print('-' * 78)

        # ---- GSD 与覆盖 ----
        print(f'{"高度(m)":>8}{"GSD(mm/px)":>12}{"单帧覆盖(m)":>16}{"每公顷张数":>12}')
        for H in HEIGHTS:
            g = gsd_mm(cam, H)
            w, h = footprint_m(cam, H)
            print(f'{H:>8}{g:>12.2f}{f"{w:.1f} x {h:.1f}":>16}{frames_per_ha(cam, H):>12.0f}')

        # ---- 病斑尺寸 -> 高度上限 ----
        print(f'\n  要使病斑脱离 small 区间（达到 {SMALL_PX} px 边长）所需的最大高度：')
        print(f'{"病斑直径":>10}{"最大高度(m)":>14}{"该高度GSD":>12}{"每公顷张数":>12}')
        for d in LESION_MM:
            H = max_height_for(cam, d, SMALL_PX)
            print(f'{f"{d} mm":>10}{H:>14.2f}{gsd_mm(cam, H):>12.3f}'
                  f'{frames_per_ha(cam, H):>12.0f}')

        # ---- 常用巡查高度下病斑落在哪一档 ----
        print(f'\n  典型巡查高度下各尺寸病斑的分档：')
        hdr = ''.join(f'{f"{d}mm":>10}' for d in LESION_MM)
        print(f'{"高度(m)":>8}{hdr}')
        for H in [2, 5, 10, 20, 30, 50]:
            g = gsd_mm(cam, H)
            row = ''.join(f'{bucket(d / g):>10}' for d in LESION_MM)
            print(f'{H:>8}{row}')

    # ---------------- 反向核算：给定高度能看见多大的目标 ----------------
    print('\n' + '=' * 78)
    print('反向核算：各高度下的最小可分辨目标尺寸')
    print('=' * 78)
    print('  这是采集方案真正要面对的问题——高度定了，能看到的最小特征有多大。')
    print(f'\n{"相机":<34}{"高度":>7}{"GSD":>10}{"达32px需":>12}{"达96px需":>12}')
    print('-' * 78)
    for name, cam in CAMERAS.items():
        for H in [10, 30, 50]:
            g = gsd_mm(cam, H)
            print(f'{name[:33]:<34}{f"{H}m":>7}{f"{g:.2f}mm":>10}'
                  f'{f"{g * SMALL_PX / 10:.1f}cm":>12}{f"{g * LARGE_PX / 10:.1f}cm":>12}')
    print('-' * 78)
    print('  即：30 m 高度下，只有 17 cm 以上的目标才够得上 medium 区间。')
    print('     病斑（毫米级）在任何常规巡查高度上都不可分辨——这是光学结论，与模型无关。')

    # ---------------- 两级航线的定量依据 ----------------
    print('\n' + '=' * 78)
    print('两级航线设计的定量依据')
    print('=' * 78)
    cam = CAMERAS['DJI Mini 4 Pro (1/1.3", 48MP)']
    d = 5
    H_small = max_height_for(cam, d, SMALL_PX)
    H_large = max_height_for(cam, d, LARGE_PX)
    print(f'  以 {d} mm 病斑、Mini 4 Pro 为例：')
    print(f'    要脱离 small（>={SMALL_PX}px）        高度须 <= {H_small:.2f} m')
    print(f'    要进入 large （>={LARGE_PX}px）        高度须 <= {H_large:.2f} m')
    print(f'    30 m 常规巡查高度下 GSD = {gsd_mm(cam, 30):.1f} mm/px，'
          f'{d} mm 病斑仅 {d / gsd_mm(cam, 30):.2f} px —— 不可见')
    print()
    print('  结论：单级航线无法同时满足覆盖效率与病斑可分辨性。')
    print(f'    第一级  30-50 m 全园普查，目标为整株/冠层级异常（米级目标，落 large）')
    print(f'    第二级  仅对一级标出的异常区域，降至 {H_small:.1f} m 以下精查')
    print()
    print('  重要提示：上述高度是"病斑达到 32 px"的光学下限，属必要条件而非充分条件。')
    print('    该高度下的实际检测精度必须用真实航拍数据测定——')
    print('    本项目现有全部实验均为地面近摄图像，不能外推至航拍视角。')
    print('=' * 78)


if __name__ == '__main__':
    main()
