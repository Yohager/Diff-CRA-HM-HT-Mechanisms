from matplotlib import pyplot as plt 

def plot_fig_homo(xvals,data,xlabel,axtitles,title):
    plt.style.use('seaborn')
    y_label = ['social cost','total payment']
    fig,subs = plt.subplots(nrows=2,ncols=3,figsize=(24,10))
    for idx in range(3):
        avg1,avg2,avg3 = data[idx][0][0][0],data[idx][0][1][0], data[idx][0][2][0]
        lb1,lb2,lb3 = data[idx][0][0][1],data[idx][0][1][1], data[idx][0][2][1]
        ub1,ub2,ub3 = data[idx][0][0][2],data[idx][0][1][2], data[idx][0][2][2]
        subs[0][idx].plot(xvals,avg1,color='#A2142F')
        subs[0][idx].plot(xvals,avg2,color='#0072BD')
        subs[0][idx].plot(xvals,avg3,color='#77AC30')
        subs[0][idx].fill_between(xvals,ub1,lb1,facecolor='#A2142F',alpha=0.35)
        subs[0][idx].fill_between(xvals,ub2,lb2,facecolor='#0072BD',alpha=0.35) 
        subs[0][idx].fill_between(xvals,ub3,lb3,facecolor='#77AC30',alpha=0.35)
        subs[0][idx].set_xlabel(xlabel,fontsize=16)
        subs[0][idx].set_ylabel(y_label[0],fontsize=16)
        subs[0][idx].legend(('ND-VCG','Diff-CRA-HM','D-VCG'),loc=4)
        subs[0][idx].set_title(axtitles[idx],fontsize=16)
        subs[0][idx].tick_params(axis='x',labelsize=16)
        subs[0][idx].tick_params(axis='y',labelsize=16)
 
    for idx in range(3):
        avg1,avg2,avg3 = data[idx][1][0][0],data[idx][1][1][0], data[idx][1][2][0]
        lb1,lb2,lb3 = data[idx][1][0][1],data[idx][1][1][1], data[idx][1][2][1]
        ub1,ub2,ub3 = data[idx][1][0][2],data[idx][1][1][2], data[idx][1][2][2]
        subs[1][idx].plot(xvals,avg1,color='#A2142F')
        subs[1][idx].plot(xvals,avg2,color='#0072BD')
        subs[1][idx].plot(xvals,avg3,color='#77AC30')
        subs[1][idx].fill_between(xvals,ub1,lb1,facecolor='#A2142F',alpha=0.35)
        subs[1][idx].fill_between(xvals,ub2,lb2,facecolor='#0072BD',alpha=0.35) 
        subs[1][idx].fill_between(xvals,ub3,lb3,facecolor='#77AC30',alpha=0.35)
        subs[1][idx].set_xlabel(xlabel,fontsize=16)
        subs[1][idx].set_ylabel(y_label[1],fontsize=16)
        subs[1][idx].legend(('ND-VCG','Diff-CRA-HM','D-VCG'),loc=4)
        subs[1][idx].tick_params(axis='x',labelsize=16)
        subs[1][idx].tick_params(axis='y',labelsize=16)
        # subs[1][idx].set_title(axtitles[idx],fontsize=16)
    plt.savefig('./new_plots/' + 'homo_' + title + '.pdf',bbox_inches='tight')
    plt.show()

def plot_fig_hete(xvals,data,budgs,xlabel,axtitles, title):
    plt.style.use('seaborn')
    # xlabel = 'probs for generating random graphs'
    ylabel1 = 'social cost'
    ylabel2 = 'total payment'
    # axtitles = ['tasks:100,prob=0.05','tasks:1000,suppliers:1000','tasks:2000,suppliers:1000']
    fig,subs = plt.subplots(nrows=2,ncols=3,figsize=(24,10))
    # ax0,1,2 record social cost 
    # for idx,ax,data in enumerate(zip(subs[:3],[data1,data2,data3])):
    for idx in range(3):
        avg1,avg2 = data[idx][0][0][0],data[idx][0][1][0]
        lb1,lb2 = data[idx][0][0][1], data[idx][0][1][1]
        ub1,ub2 = data[idx][0][0][2], data[idx][0][1][2]
        subs[0][idx].plot(xvals,budgs[idx],color='#77AC30',linewidth=4)
        subs[0][idx].plot(xvals,avg1,color='#A2142F')
        subs[0][idx].plot(xvals,avg2,color='#0072BD')
        subs[0][idx].fill_between(xvals,ub1,lb1,facecolor='#A2142F',alpha=0.35)
        subs[0][idx].fill_between(xvals,ub2,lb2,facecolor='#0072BD',alpha=0.35)
        # subs[0][idx].set_xticks(fontsize=18)
        # subs[0][idx].set_yticks(fontsize=18)
        subs[0][idx].set_xlabel(xlabel,fontsize=16)
        subs[0][idx].set_ylabel(ylabel1,fontsize=16)
        subs[0][idx].legend(('Budget','Greedy','Diff-CRA-HT'),loc=4)
        subs[0][idx].set_title(axtitles[idx],fontsize=16)
        subs[0][idx].tick_params(axis='x',labelsize=16)
        subs[0][idx].tick_params(axis='y',labelsize=16)
    # ax3,4,5 record payment 
    # for idx,ax,data in enumerate(zip(subs[3:],[data1,data2,data3])):
    for idx in range(3):
        avg1,avg2 = data[idx][1][0][0],data[idx][1][1][0]
        lb1,lb2 = data[idx][1][0][1], data[idx][1][1][1]
        ub1,ub2 = data[idx][1][0][2], data[idx][1][1][2]
        subs[1][idx].plot(xvals,budgs[idx],color='#77AC30',linewidth=4)
        subs[1][idx].plot(xvals,avg1,color='#A2142F')
        subs[1][idx].plot(xvals,avg2,color='#0072BD')
        subs[1][idx].fill_between(xvals,ub1,lb1,facecolor='#A2142F',alpha=0.35)
        subs[1][idx].fill_between(xvals,ub2,lb2,facecolor='#0072BD',alpha=0.35)
        # subs[0][idx].set_xticks(fontsize=18)
        # subs[0][idx].set_yticks(fontsize=18)
        subs[1][idx].set_xlabel(xlabel,fontsize=16)
        subs[1][idx].set_ylabel(ylabel2,fontsize=16)
        subs[1][idx].legend(('Budget','Greedy','Diff-CRA-HT'),loc=4)
        subs[1][idx].tick_params(axis='x',labelsize=16)
        subs[1][idx].tick_params(axis='y',labelsize=16)
        # subs[1][idx].set_title(axtitles[idx],fontsize=16)
    # fig.show()
    plt.savefig('./new_plots/' + 'hete_' + title + '.pdf',bbox_inches='tight')
    plt.show()