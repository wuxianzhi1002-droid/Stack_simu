import os
import shutil

def install_assets():
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 源文件路径 (现在在 01_Material_Assets 目录下)
    src_agf = os.path.join(base_dir, "01_Material_Assets", "STACK_MATERIALS.AGF")
    src_dat = os.path.join(base_dir, "01_Material_Assets", "COATING.DAT")
    
    # 目标路径 (基于用户提供的 Zemax 数据目录)
    dest_root = r"D:\Users\wuxianzhi\Documents\Zemax"
    dest_agf_dir = os.path.join(dest_root, "Glasscat")
    dest_dat_dir = os.path.join(dest_root, "Coatings")
    
    print("--- 正在安装 Zemax 仿真资产 ---")
    
    # 1. 安装 AGF (玻璃库)
    if os.path.exists(src_agf):
        os.makedirs(dest_agf_dir, exist_ok=True)
        dest_agf_path = os.path.join(dest_agf_dir, "STACK_MATERIALS.AGF")
        shutil.copy2(src_agf, dest_agf_path)
        print(f"[成功] 玻璃库已安装至: {dest_agf_path}")
    else:
        print(f"[错误] 找不到源文件: {src_agf}")

    # 2. 安装/合并 COATING.DAT
    if os.path.exists(src_dat):
        os.makedirs(dest_dat_dir, exist_ok=True)
        dest_dat_path = os.path.join(dest_dat_dir, "COATING.DAT")
        
        # 读取源文件 (UTF-8)
        with open(src_dat, 'r', encoding='utf-8') as f_src:
            new_coatings = f_src.read()
            
        # 检查是否已经存在
        if os.path.exists(dest_dat_path):
            # Zemax 的 COATING.DAT 经常使用 utf-16
            try:
                with open(dest_dat_path, 'r', encoding='utf-16') as f_dest:
                    existing_content = f_dest.read()
            except:
                with open(dest_dat_path, 'r', encoding='utf-8', errors='ignore') as f_dest:
                    existing_content = f_dest.read()

            if "COAT PSS_TiO2" in existing_content:
                print("[提示] 膜层定义已存在于 COATING.DAT 中，跳过合并。")
            else:
                # 保持追加，尝试使用 utf-16 追加，如果失败则用 utf-8
                try:
                    with open(dest_dat_path, 'a', encoding='utf-16') as f_append:
                        f_append.write("\n" + new_coatings)
                except:
                    with open(dest_dat_path, 'a', encoding='utf-8') as f_append:
                        f_append.write("\n" + new_coatings)
                print(f"[成功] 膜层定义已追加至: {dest_dat_path}")
        else:
            shutil.copy2(src_dat, dest_dat_path)
            print(f"[成功] 膜层文件已创建: {dest_dat_path}")
    else:
        print(f"[错误] 找不到源文件: {src_dat}")

if __name__ == "__main__":
    install_assets()
