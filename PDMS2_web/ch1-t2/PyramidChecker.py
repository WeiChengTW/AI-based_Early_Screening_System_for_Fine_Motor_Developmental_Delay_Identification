import numpy as np

class PyramidCheck:
    @staticmethod
    def is_pyramid_shape(layers, x_threshold):
        """
        檢查分層是否符合金字塔形狀（1-2-3結構且垂直對齊）
        
        參數:
        - layers: 已按 Y 座標排序的分層 [[(x,y)], [(x,y),(x,y)], ...]
        - x_threshold: X 座標對齊的容許誤差
        """
        if len(layers) != 3:
            print(f"❌ 層數不對: {len(layers)}（期望 3 層）")
            return False

        # 檢查每層數量
        layer_sizes = [len(layer) for layer in layers]
        if layer_sizes != [1, 2, 3]:
            print(f"❌ 層級結構錯誤: {layer_sizes}（期望 [1,2,3]）")
            return False

        # 計算各層平均 X 座標
        avg_xs = []
        for i, layer in enumerate(layers):
            x_coords = [p[0] for p in layer if isinstance(p, (list, tuple)) and len(p) == 2]
            if not x_coords:
                print(f"❌ 第 {i} 層無法計算座標")
                return False
            avg_x = np.mean(x_coords)
            avg_xs.append(avg_x)
            print(f"第 {i} 層: {len(layer)} 個方塊, 平均 X = {avg_x:.2f}")

        # 檢查垂直對齊
        max_diff = max(
            abs(avg_xs[0] - avg_xs[1]),
            abs(avg_xs[1] - avg_xs[2]),
            abs(avg_xs[0] - avg_xs[2])
        )
        
        is_aligned = max_diff <= x_threshold
        
        print(f"X 座標: {[f'{x:.1f}' for x in avg_xs]}")
        print(f"最大偏差: {max_diff:.2f}, 閾值: {x_threshold:.2f}")
        print(f"對齊判定: {'✅ PASS' if is_aligned else '❌ FAIL'}")
        
        return is_aligned

    def check_pyramid(self, layers, block_width, gap):
        """
        完整金字塔檢查（結構 + 空隙）
        """
        if len(layers) != 3:
            return False, f"層數錯誤: {len(layers)}"

        # 計算 X 閾值（放寬到 80%）
        x_threshold = block_width * 0.8
        
        # 檢查結構
        is_valid_structure = self.is_pyramid_shape(layers, x_threshold)
        
        if not is_valid_structure:
            return False, "Not Pyramid", 0

        if not gap:
            return True, "Pyramid!, But no gap", 1
        
        return True, "Pyramid!", 2