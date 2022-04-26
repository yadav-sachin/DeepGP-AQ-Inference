import os
import re

script_dir = os.path.dirname(__file__)

for item_name in os.listdir(script_dir):
    print(item_name)
    if ".png" in item_name:
        num = re.findall(r'\d+', item_name)[-1]
        str_num = str(num)
        while (len(str_num)) < 3:
            str_num = "0" + str_num
        new_name = f"deep_variational_station_1006_{str_num}.png"
        old_path = os.path.join(script_dir, item_name)
        new_path = os.path.join(script_dir, new_name)
        os.rename(old_path, new_path)
        print(new_name)