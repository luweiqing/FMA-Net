"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.
Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
Modified by Xingyi Zhou
"""

import argparse
import glob
import os
import logging
import motmetrics as mm
import pandas as pd
from collections import OrderedDict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="""
Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...
Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...
Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--result', type=str, help='Log level', default='./SatMTB/DLADCN/results/tracking*')
    parser.add_argument('--cate', type=str, help='choose cate from airplane, car, ship and train', default='car')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mtb-sat')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    return parser.parse_args()

NAME_LABEL = {
    'car': 0,
    'airplane':      1,
    'ship':     2,
    'train':    3
}

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logging.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.7))
            names.append(k)
        else:
            logging.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 配置日志系统
    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))        
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    # 设置LAP求解器（线性分配问题）
    if args.solver:
        mm.lap.default_solver = args.solver  # 使用指定的优化算法

    # 加载ground truth文件
    gtfiles = glob.glob(os.path.join('../MP2Net/data/SatMTB/test/label', args.cate, '*'))
    # 加载测试结果文件
    tsfiles = [f for f in glob.glob(os.path.join(args.result, args.cate, '*.txt'))]

    # 打印文件统计信息
    logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logging.info('Loading files.')
    
    # 加载并解析ground truth数据
    gt = OrderedDict([
        (os.path.splitext(Path(f).parts[-1])[0],  # 提取文件名作为键
         mm.io.loadtxt(f, NAME_LABEL[args.cate], fmt=args.fmt))  # 加载为DataFrame
        for f in gtfiles
    ])
    
    # 加载并解析测试结果数据
    ts = OrderedDict([
        (os.path.splitext(Path(f).parts[-1])[0], 
         mm.io.loadtxt(f, NAME_LABEL[args.cate], fmt='mot16'))  # 使用MOT16格式解析
        for f in tsfiles
    ])
    
    # 创建指标计算器
    mh = mm.metrics.create()    
    
    # 比较ground truth和测试结果
    accs, names = compare_dataframes(gt, ts)
    
    # 定义需要计算的指标
    logging.info('Running metrics')
    metrics = [
        'recall',        # 召回率
        'precision',     # 精确率
        'num_unique_objects',  # 唯一目标数量
        'mostly_tracked',      # 主要被跟踪的目标比例
        'partially_tracked',   # 部分被跟踪的目标比例
        'mostly_lost',         # 主要丢失的目标比例
        'num_false_positives', # 误报数量
        'num_misses',          # 漏检数量
        'num_switches',        # ID切换次数
        'num_fragmentations',  # 跟踪中断次数
        'mota',          # 多目标跟踪准确率
        'motp',          # 多目标跟踪精确度
        'num_objects'    # 总目标数
    ]
    
    # 计算指标
    summary = mh.compute_many(
        accs, 
        names=names, 
        metrics=metrics, 
        generate_overall=True  # 生成总体统计
    )
    
    # 将部分指标转换为比例
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 
          'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 
          'mostly_lost']
    }
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    
    # 设置数值格式化方式
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 
      'num_fragmentations', 'mostly_tracked', 'partially_tracked', 
      'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']  # 使用百分比格式
    
    # 重新计算包含所有标准MOT指标的汇总
    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(
        accs, 
        names=names, 
        metrics=metrics, 
        generate_overall=True
    )
    
    # 打印最终结果
    print(mm.io.render_summary(
        summary, 
        formatters=mh.formatters,  # 使用格式化器
        namemap=mm.io.motchallenge_metric_names  # 指标名称映射
    ))

    logging.info('Completed')

