import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os, argparse
import numpy as np

## avoid type 3 font in plot, use 42 instead
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def load_data(path):
    data = pd.read_csv(path)
    print(data.head(2))
    return data


def initiation_performance(data=None,outputpath:str=None):
    """
    plot difference of benign and random initiation
    :param data:
    :param outputpath:
    :return:
    """
    marker_size = 18
    font_size = 18

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    plt.figure(figsize=(8,6),dpi=150)
    plt.plot(data['payload_size'], data['benign'], color='red', marker='*', linestyle='--', linewidth=2, markeredgewidth=2,
             fillstyle='none', markersize=marker_size, label='Benign Initiation')
    plt.plot(data['payload_size'], data['random'], color='blue', marker='^', linestyle='--', linewidth=2, markeredgewidth=2,
             fillstyle='none', markersize=marker_size, label='Random Initiation')
    plt.xlabel('Crafted Byte Ratio', {'size': font_size})
    plt.ylabel('Evasion Rate', {'size': font_size})
    plt.xticks(data['payload_size'])
    plt.tick_params(labelsize=font_size)
    plt.legend(loc='best', fontsize=font_size)
    plt.savefig(outputpath + 'random_benign_initiation.eps')
    plt.show()


def different_perturb_place(data=None,outputpath=None):
    """
    impact of different perturbation spaces
    :param data:
    :param outputpath:
    :return:
    """

    marker_size = 18
    font_size = 18

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(data['payload_size'], data['PartialDos'], color='red', marker='+', linestyle='--', linewidth=2,
             markeredgewidth=2,fillstyle='none', markersize=marker_size, label='PD')
    plt.plot(data['payload_size'], data['Slack'], color='blue', marker='^', linestyle='--', linewidth=2,
             markeredgewidth=2,fillstyle='none', markersize=marker_size, label='S')
    plt.plot(data['payload_size'], data['FullDos'], color='green', marker='1', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='FD')
    plt.plot(data['payload_size'], data['ContentShift'], color='yellow', marker='o', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='CS')
    plt.plot(data['payload_size'], data['PartialDosContentShift'], color='grey', marker='>', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='PD+CS')
    plt.plot(data['payload_size'], data['FullDosSlack'], color='black', marker='<', linestyle='--',
             linewidth=2,markeredgewidth=2, fillstyle='none', markersize=marker_size, label='FD+S')
    plt.plot(data['payload_size'], data['PartialDosSlack'], color='pink', marker='.', linestyle='--',
             linewidth=2,markeredgewidth=2, fillstyle='none', markersize=marker_size, label='PD+S')
    plt.plot(data['payload_size'], data['ContentShiftSlack'], color='orange', marker='*', linestyle='--',
             linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size, label='CS+S')
    plt.plot(data['payload_size'], data['PartialDosContentShiftSlack'], color='purple', marker='2', linestyle='--',
             linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size, label='PD+CS+S')
    plt.plot(data['payload_size'], data['DosExtend'], color='brown', marker='3', linestyle='--',
             linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size, label='DE')
    plt.plot(data['payload_size'], data['FullDosContentShift'], color='olive', marker='4', linestyle='--',
             linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size, label='FD+CS')
    plt.plot(data['payload_size'], data['DosExtendContentShift'], color='cyan', marker='s', linestyle='--',
             linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size, label='DE+CS')
    plt.plot(data['payload_size'], data['DosExtendSlack'], color='rosybrown', marker='p', linestyle='--',
             linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size, label='DE+S')
    plt.plot(data['payload_size'], data['FullDosContentShiftSlack'], color='goldenrod', marker='h', linestyle='--',
             linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size, label='FD+CS+S')
    plt.plot(data['payload_size'], data['DosExtendContentShiftSlack'], color='darkcyan', marker='x', linestyle='--',
             linewidth=2, markeredgewidth=2, fillstyle='none', markersize=marker_size, label='DE+CS+S')
    plt.xlabel('Payload Size', {'size': font_size})
    plt.ylabel('Evasion Rate', {'size': font_size})
    plt.xticks(data['payload_size'])
    plt.ylim(0,1.1)
    plt.tick_params(labelsize=font_size)
    plt.legend(loc='best', fontsize=13)
    plt.savefig(outputpath + 'different_perturb_place.eps')
    plt.show()



def different_adversary_strength(data=None,outputpath=None):
    """
    impact of different adversary strength
    :param data:
    :param outputpath:
    :return:
    """

    marker_size = 18
    font_size = 18

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(data['payload_size'], data['0.1'], color='orange', marker='+', linestyle='--', linewidth=2,
             markeredgewidth=2,fillstyle='none', markersize=marker_size, label='$ϵ$=0.1')
    plt.plot(data['payload_size'], data['0.2'], color='blue', marker='^', linestyle='--', linewidth=2,
             markeredgewidth=2,fillstyle='none', markersize=marker_size, label='$ϵ$=0.2')
    plt.plot(data['payload_size'], data['0.3'], color='green', marker='1', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='$ϵ$=0.3')
    plt.plot(data['payload_size'], data['0.4'], color='red', marker='*', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='$ϵ$=0.4')
    plt.plot(data['payload_size'], data['0.5'], color='grey', marker='3', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='$ϵ$=0.5')
    plt.plot(data['payload_size'], data['0.6'], color='olive', marker='2', linestyle='--',
             linewidth=2,markeredgewidth=2, fillstyle='none', markersize=marker_size, label='$ϵ$=0.6')
    plt.plot(data['payload_size'], data['0.7'], color='pink', marker='.', linestyle='--',
             linewidth=2,markeredgewidth=2, fillstyle='none', markersize=marker_size, label='$ϵ$=0.7')
    plt.xlabel('Crafted Byte Ratio', {'size': font_size})
    plt.ylabel('Evasion Rate', {'size': font_size})
    plt.xticks(data['payload_size'])
    # plt.ylim(0,1)
    plt.tick_params(labelsize=font_size)
    plt.legend(loc='best', fontsize=font_size)
    plt.savefig(outputpath + 'different_adversary_strength.eps')
    plt.show()



def different_input_size(data=None, outputpath=None):
    """
    impact of different input size
    :param data:
    :param outputpath:
    :return:
    """

    marker_size = 18
    font_size = 18

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(data['payload_size'], data['102400'], color='red', marker='*', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='$d$=102400')
    plt.plot(data['payload_size'], data['204800'], color='blue', marker='^', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='$d$=204800')
    plt.plot(data['payload_size'], data['409600'], color='orange', marker='+', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='$d$=409600')
    plt.xlabel('Crafted Byte Ratio', {'size': font_size})
    plt.ylabel('Evasion Rate', {'size': font_size})
    plt.xticks(data['payload_size'])
    # plt.ylim(0,1)
    plt.tick_params(labelsize=font_size)
    plt.legend(loc='best', fontsize=font_size)
    plt.savefig(outputpath + 'different_input_size.eps')
    plt.show()


def time_overhead(data=None, outputpath=None):
    """
    time overhead across different payload size under multiple malware detectors (1,2,3)
    :param data:
    :param outputpath:
    :return:
    """
    font_size = 18
    BarWidth = 0.25

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    plt.subplots(figsize=(8,6),dpi=150)

    ## set position of bar on X axis
    br1 = np.arange(len(data['payload_size']))
    br2 = [x + BarWidth for x in br1]
    br3 = [x + BarWidth for x in br2]

    ## plot
    plt.bar(br1,data['one'],color='grey',width=BarWidth,label='One Malware Detector',hatch='/')
    plt.bar(br2,data['two'],color='cornflowerblue',width=BarWidth,label='Two Malware Detectors',hatch='-')
    plt.bar(br3,data['three'],color='orange',width=BarWidth,label='Three Malware Detectors')

    ## add Xticks
    plt.xlabel('Crafted Byte Ratio', fontsize=font_size)
    plt.ylabel('Generation Time (second/sample)', fontsize=font_size)
    plt.xticks([r + BarWidth for r in range(len(data['payload_size']))],
               ['0.5%', '1%', '2%', '4%', '8%','16%','32%'])
    plt.tick_params(labelsize=font_size)
    plt.legend(loc='best',fontsize=font_size)
    plt.savefig(outputpath + 'time_overhead_different_num_detectors.eps')
    plt.show()


def time_overhead_different_adversary(data=None, outputpath=None):
    """
    time overhead across different payload size with different adversary (FGSM,PGD,FFGSM,CW)
    :param data:
    :param outputpath:
    :return:
    """
    font_size = 18
    BarWidth = 0.25

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    plt.subplots(figsize=(8,6),dpi=150)

    ## set position of bar on X axis
    br1 = np.arange(len(data['payload_size']))
    br2 = [x + BarWidth for x in br1]
    br3 = [x + BarWidth for x in br2]

    ## plot
    plt.bar(br1,data['FGSM'],color='grey',width=BarWidth,label='FGSM',hatch='/')
    plt.bar(br2,data['FFGSM'],color='cornflowerblue',width=BarWidth,label='FFGSM',hatch='-')
    plt.bar(br3,data['PGD'],color='orange',width=BarWidth,label='PGD')

    ## add Xticks
    plt.xlabel('Crafted Byte Ratio', fontsize=font_size)
    plt.ylabel('Generation Time (second/sample)', fontsize=font_size)
    plt.xticks([r + BarWidth for r in range(len(data['payload_size']))],
               ['0.5%', '1%', '2%', '4%', '8%','16%','32%'])
    plt.tick_params(labelsize=font_size)
    plt.legend(loc='best',fontsize=font_size)
    plt.savefig(outputpath + 'time_overhead_different_adversary.eps')
    plt.show()



def different_dataset(data=None, outputpath=None):
    """
    impact of different dataset
    :param data:
    :param outputpath:
    :return:
    """
    marker_size = 18
    font_size = 18

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(data['payload_size'], data['self-collected'], color='red', marker='*', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='ME')
    plt.plot(data['payload_size'], data['phd'], color='blue', marker='^', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='phd')
    plt.xlabel('Crafted Byte Ratio', {'size': font_size})
    plt.ylabel('Evasion Rate', {'size': font_size})
    plt.xticks(data['payload_size'])
    plt.ylim(0,1.1)
    plt.tick_params(labelsize=font_size)
    plt.legend(loc='best', fontsize=font_size)
    plt.savefig(outputpath + 'different_dataset.eps')
    plt.show()


def different_adversary(data=None, outputpath=None):
    """
    impact of different dataset
    :param data:
    :param outputpath:
    :return:
    """
    marker_size = 18
    font_size = 18

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(data['payload_size'], data['FGSM'], color='red', marker='*', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='FGSM')
    plt.plot(data['payload_size'], data['PGD'], color='blue', marker='^', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='PGD')
    plt.plot(data['payload_size'], data['FFGSM'], color='orange', marker='+', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='FFGSM')
    plt.xlabel('Crafted Byte Ratio', {'size': font_size})
    plt.ylabel('Evasion Rate', {'size': font_size})
    plt.xticks(data['payload_size'])
    plt.ylim(0,1.1)
    plt.tick_params(labelsize=font_size)
    plt.legend(loc='best', fontsize=font_size)
    plt.savefig(outputpath + 'different_adversary.eps')
    plt.show()

def evasion_performance(data=None, outputpath=None):
    """
    evasion performance against two malware detectors (both with raw bytes input and one with raw byte, one with image),
    against three detectors
    :param data:
    :param outputpath:
    :return:
    """
    marker_size = 18
    font_size = 18

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(data['payload_size'], data['two_detectors_raw_bytes'], color='red', marker='*', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='MalConv and FireEyeNet')
    plt.plot(data['payload_size'], data['two_detectors_different_input_format'], color='blue', marker='^', linestyle='--', linewidth=2,
             markeredgewidth=2, fillstyle='none', markersize=marker_size, label='MalConv and ResNet18')
    # plt.plot(data['payload_size'], data['three_detectors'], color='orange', marker='+', linestyle='--', linewidth=2,
    #          markeredgewidth=2, fillstyle='none', markersize=marker_size, label='Three Malware Detectors (raw bytes)')
    plt.xlabel('Crafted Byte Ratio', {'size': font_size})
    plt.ylabel('Evasion Rate', {'size': font_size})
    plt.xticks(data['payload_size'])
    plt.ylim(0, 1.1)
    plt.tick_params(labelsize=font_size)
    plt.legend(loc='lower right', fontsize=font_size)
    plt.savefig(outputpath + 'evasion_performance.eps')
    plt.show()


def main():
    data = load_data(args.input_path)

    ## different initiation
    if args.plot_name=='different_initiation':
        initiation_performance(data,args.output_path)

    ## different perturb place
    elif args.plot_name=='different_perturb_place':
        different_perturb_place(data,args.output_path)

    ## different adversary strength
    elif args.plot_name == 'different_adversary_strength':
        different_adversary_strength(data,args.output_path)

    ## different input size
    elif args.plot_name == 'different_input_size':
        different_input_size(data,args.output_path)

    ## time overhead
    elif args.plot_name == 'time_overhead_different_num_detectors':
        time_overhead(data,args.output_path)

    ## different dataset
    elif args.plot_name == 'different_dataset':
        different_dataset(data,args.output_path)

    ## different adversary
    elif args.plot_name == 'different_adversary':
        different_adversary(data,args.output_path)

    ## time overhead with different adversary
    elif args.plot_name == 'time_overhead_different_adversary':
        time_overhead_different_adversary(data,args.output_path)

    ## evasion performance against two/three detectors (with same/different input format)
    elif args.plot_name == 'evasion_performance':
        evasion_performance(data,args.output_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='figure plot', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path',default=None,type=str,help='csv input path')
    parser.add_argument('--plot_name',default=None,type=str,help='plot type name')
    parser.add_argument('--output_path',default='../result/figure/',type=str,help='folder for save figure, e.g., ../result/figure/')

    args = parser.parse_args()
    print('\n',args,'\n')

    main()
