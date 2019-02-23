# -*- coding: utf-8 -*-
import os
import requests
import hashlib
import time
from random import random
from datetime import datetime
import logging

today = datetime.today().date()

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='log/%s.log' % today,
    filemode='a')

def save_img(file, src):
    '''
    This function is used to save pictures.
    Initiates an HTTP request to the picture URL,
    gets the binary code,
    writes the code to the local file,
    and completes the preservation of a picture.
    :param file:folder path
    :param src: image url
    :return:
    '''
    if os.path.exists(file):
        print(f'-{file}已存在，跳过。-')
    else: # This is done simply to dedup process
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}
            res = requests.get(src, timeout=3, verify=False,headers=headers)
            # print(res.content)
        except Exception as e:
            print(f'--{e}--')
            logging.warning(f'{os.path.split(__file__)[1]} - {src} - {e}')
            return False
        else:
            if res.status_code == 200:
                img = res.content
                open(file, 'wb').write(img)
                time.sleep(random())
                return True


def path_processor(site, folder):
    '''
    :param site: site name，pexels，pixabay，google
    :param folder: category name
    :return path: path
    '''
    categories = [['乐器', '笛子', '鼓', '长号', '钢琴', '小提琴','女脸'],
                  ['交通工具', '面包车', '摩托车', '轿车', 'SUV', '电瓶车', '三轮车', '自行车', '船', '大客车', '微型车'],
                  ['办公产品', '显示屏', '鼠标', '垃圾篓', '路由器', '折叠床', '办公桌', '电话', '打印机', '键盘', '书本', '电脑椅', '投影仪', '绿植盆栽', '本子',
                   '笔类', '接线板', '笔记本电脑', '文件收纳', '多肉盆栽', '文件柜', '碎纸机', '平板电脑', '订书机', '保险柜', '计算器'],
                  ['场景', '商场内景', '酒吧夜店', '卧室', '湖泊', '山', '地铁内景', '厢式电梯外景', '沙滩', '轿车内景', '篮球场', '图书馆内景', '跑道', '广场',
                   '客厅',
                   '田野', '公路', '卫生间', '超市内景', '大门口', '街道', '电影院内景', '草坪', '厨房', '厢式电梯内景', '写字楼外景', '瀑布', '足球场', '鲜花',
                   '天空',
                   '办公室', '树木', '手扶电梯', '餐厅内景', '健身房内景'],
                  ['家用电器', '洗衣机', '壁挂空调', '电磁炉', '超薄电视', '微波炉', '吸尘器', '电饭煲', '加湿器', '电热片', '燃气灶', '电风扇', '柜式空调', '咖啡机',
                   '榨汁机', '剃须刀', '扫地机器人', '面包机', '电水壶', '电吹风', '冰箱', '饮水机', '熨斗', '油烟机'],
                  ['数码产品', '手机', '音箱', '相机', 'VR眼镜', '三脚架', '体感车', '手表', '无人机', '耳机耳麦'],
                  ['服饰', '短裤', '连衣裙', '休闲裤', '衬衫', '运动鞋', '外套', 'T恤', '凉鞋', '皮鞋', '牛仔裤', '拖鞋'],
                  ['活动', '运动会', '婚礼', '聚餐'],
                  ['生活用品', '玻璃杯', '碗', '运动水壶', '保鲜盒', '锅具', '瓜果刨', '菜刀', '剪刀', '筷子', '叉', '椅子', '梯子', '沙发', '马克杯', '衣架',
                   '盘子', '伞', '勺子', '餐桌'],
                  ['箱包装饰', '双肩包', '化妆品', '珠宝', '女式挎包', '眼镜', '拉杆箱', '手提包', '钱包', '腰带'],
                  ['食品', '车厘子 樱桃', '三文鱼', '火锅', '矿泉水', '休闲零食', '火龙果', '香蕉', '椰子', '鱿鱼 章鱼', '面包', '饼干', '烧烤',
                   '糖果 巧克力',
                   '海参', '坚果炒货', '贝类', '海产干货', '鸡翅', '牛奶', '芒果', '食用油', '猕猴桃', '牛排', '虾类', '蛋糕', '橙子', '西餐', '饮料',
                   '方便面',
                   '鱼类', '膨化食品', '牛油果', '小龙虾', '米面', '蓝莓', '菠萝', '红酒', '咖啡粉', '咖啡豆', '榴莲', '白酒', '苹果', '肉', '蟹类']]
    for cat in categories:
        if folder in cat:
            path = f'{site}/{cat[0]}/{folder}/'
            break
    else:
        raise NameError("Please input correct category name！")
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def img_name_processor(src):
    """
    This function is used to handle the file name of the saved picture.
    Hash the URL of the picture as its filename.
    :param src: image url
    :return: image filename
    """
    h5 = hashlib.md5()
    h5.update(src.encode('utf-8'))
    img = h5.hexdigest() + '.jpg'
    return img


if __name__ == "__main__":
    save_img('test.jpg','https://images.pexels.com/photos/458766/pexels-photo-458766.jpeg?auto=compress&cs=tinysrgb&h=350')
