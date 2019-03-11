import os
import numpy as np
import re
import tqdm
import RasterProcess as Rc
import pandas as pd

class StationProcessing(object):
    """
    # 0 Bj 1 DB 2 TQ
    """

    @classmethod
    def local2station(cls, l1, l2, l3):
        """
        :param l1: bj
        :param l2: tq
        :param l3: db
        :return:
        """
        fl1 = open(l1, 'r').readlines()
        fl2 = open(l2, 'r').readlines()
        fl3 = open(l3, 'r').readlines()

        station2fl1 = [fl1[s].split(',')[0] for s in range(1, len(fl1))]
        station2fl2 = [fl2[s].split(',')[0] for s in range(1, len(fl2))]
        station2fl3 = [fl3[s].split(',')[0] for s in range(1, len(fl3))]

        return {0: station2fl1, 1: station2fl2, 2: station2fl3}


class TextProcessing(StationProcessing):
    """
    {'tags':{'stations':[[date, snow depth]]}}
    """
    def __init__(self, l1, l2, l3):
        """
        :param l1:
        :param l2:
        :param l3:
        """
        self.station = self.local2station(l1=l1, l2=l2, l3=l3)
        self.count = 0
        self.index = 0

    def optimize_cal(self, tex):
        """
        :param tex:
        :return:
        """
        result = {}

        station = self.station
        filename = self.is2seq_snow_season(tex=tex).flatten()
        print(filename)

        """
        for key in station.keys():

            stat = station[key]

            temp = []

            for name in tqdm.trange(len(filename)):

                for stn in range(len(stat)):

                    content = self.text_content(tex=tex, name=filename[name])
                    line, col = np.shape(content)

                    temp_ = []

                    step = self.opt_step(y=str(filename[name]).split('_')[6][0:4], m=str(filename[name]).split('_')[6][4:6])

                    for sc in range(0, line, step):

                        if int(stat[stn]) == int(content[sc][0]):

                            temp_ = content[sc: sc + step, ::]

                        else:
                            continue

                    temp.append(temp_)

            result[key] = temp

        for kc in result.keys():
            if int(kc) == 0:

                content = result.get(kc)

                foo = open('./RESULT' + '/' + 'BJ.txt', 'w')

                for cn in content:
                    for cm in cn:

                        if len(cm) == 0:
                            continue

                        else:
                            foo.writelines(str(cm[0]) + ',' +
                                           str(self.reform2date(y=cm[1],
                                                                m=cm[2],
                                                                d=cm[3],
                                                                )) + ',' + str(cm[4]) + '\n')

                foo.close()

            if int(kc) == 1:

                content = result.get(kc)

                foo = open('./RESULT' + '/' + 'DB.txt', 'w')

                for cn in content:
                    for cm in cn:

                        if len(cm) == 0:
                            continue

                        else:
                            foo.writelines(str(cm[0]) + ',' +
                                           str(self.reform2date(y=cm[1],
                                                                m=cm[2],
                                                                d=cm[3],
                                                                )) + ',' + str(cm[4]) + '\n')

                foo.close()

            if int(kc) == 2:

                content = result.get(kc)

                foo = open('./RESULT' + '/' + 'QT.txt', 'w')

                for cn in content:
                    for cm in cn:

                        if len(cm) == 0:
                            continue

                        else:
                            foo.writelines(str(cm[0]) + ',' +
                                           str(self.reform2date(y=cm[1],
                                                                m=cm[2],
                                                                d=cm[3],
                                                                )) + ',' + str(cm[4]) + '\n')

                foo.close()
        """
    @classmethod
    def opt_step(cls, y, m):
        """
        :param y:
        :param m:
        :return:
        """
        mm = re.findall(pattern=r'[^0]\d*', string=m)

        if int(mm[0]) in [1, 3, 5, 7, 8, 12]:
            return 31

        elif int(mm[0]) == 2:

            if int(y) % 4 == 0 or int(y) % 400 == 0:
                return 29
            else:
                return 28
        else:
            return 30

    def cal_observe_data(self, tex):
        """
        :param tex:
        :return:
        """
        station = self.station
        filename = self.is2seq_snow_season(tex=tex)

        result = {}

        for key in station.keys():
            stat = station.get(key)
            dit = {}
            for stn in tqdm.trange(len(stat)):
                temp = []
                for name in filename.flatten():

                    content = self.text_content(tex=tex, name=name)

                    for st in range(len(content)):

                        if int(content[st][0]) == int(stat[stn]):

                            temp.append([self.reform2date(y=content[st][1],
                                                          m=content[st][2],
                                                          d=content[st][3]), content[st][4]])
                        else:
                            continue

                dit[stat[stn]] = temp

            result[key] = dit

        for kc in result.keys():
            if kc == 0:
                if os.path.exists('./RESULT/BJ'):
                    content = result.get(kc)

                    for cn in content.keys():
                        foo = open('./RESULT/BJ' + '/' + cn, 'w')

                        cd = content.get(cn)
                        for scm in cd:
                            foo.writelines(str(scm[0] + ',' + str(scm[1] + '\n')))

                        foo.close()
                else:
                    os.mkdir('./RESULT/BJ')

            if kc == 1:
                if os.path.exists('./RESULT/DB'):
                    content = result.get(kc)

                    for cn in content.keys():
                        foo = open('./RESULT/DB' + '/' + cn, 'w')

                        cd = content.get(cn)
                        for scm in cd:
                            foo.writelines(str(scm[0] + ',' + str(scm[1] + '\n')))

                        foo.close()
                else:
                    os.mkdir('./RESULT/DB')

            if kc == 2:
                if os.path.exists('./RESULT/TQ'):
                    content = result.get(kc)

                    for cn in content.keys():
                        foo = open('./RESULT/TQ' + '/' + cn, 'w')

                        cd = content.get(cn)
                        for scm in cd:
                            foo.writelines(str(scm[0] + ',' + str(scm[1] + '\n')))

                        foo.close()
                    else:
                        os.mkdir('./RESULT/TQ')

    @classmethod
    def reform2date(cls, y, m, d):
        """
        :param y:
        :param m:
        :param d:
        :return:
        """
        if (int(m) < 10) and (int(d) < 10):
            return str(y) + '0' + str(m) + '0' + str(d)

        if (int(m) < 10) and (int(d) >= 10):
            return str(y) + '0' + str(m) + str(d)

        if (int(m) >= 10) and (int(d) < 10):
            return str(y) + str(m) + '0' + str(d)

        if (int(m) >= 10) and (int(d) >= 10):
            return str(y) + str(m) + str(d)

    def is2seq_snow_season(self, tex):
        """
        :return:
        """
        date = np.array(self.get2year(tex=tex), dtype=np.str)
        filename = self.get2filename(tex=tex)

        temp_, result, result_ = [], [], []

        for sc in date:

            if self.is2calculation(month=sc[4:6]):
                temp_.append(sc[0:4])
            else:
                continue
        unique = np.unique(temp_)

        for sc in unique:
            c = np.int(sc)
            cm = np.array([str(c - 1) + str(11), str(c - 1) + str(12), str(c) + '01', str(c) + '02', str(c) + '03'], dtype=np.str)

            if (cm[0] in date) and (cm[1] in date) and (cm[2] in date) and (cm[3] in date) and (cm[4] in date):
                result.append(cm)
            else:
                continue

        def ite(a, b):
            """
            :param a:
            :param b:
            :return:
            """
            for s_ in a:
                if str(s_).split('_')[6] == b:
                    return s_
                else:
                    continue

        for sc in result:
            tmp = []
            for ll in sc:
                tmp.append(ite(a=filename, b=ll))

            result_.append(tmp)

        return np.array(result_)

    @classmethod
    def get2year(cls, tex):
        """
        :param tex:
        :return:
        """
        text = os.listdir(tex)
        date = [sc.split('_')[6] for sc in text]
        return date

    @classmethod
    def get2filename(cls, tex):
        """
        :param tex:
        :return:
        """
        return os.listdir(tex)

    @classmethod
    def is2calculation(cls, month):
        """
        :param month:
        :return:
        """

        if np.int(month[0]) == 0 and np.int(month[1]) <= 3:
            return True

        elif np.int(month[0]) != 0 and np.int(month) >= 11:
            return True

        else:
            return False

    @classmethod
    def text_content(cls, tex, name):
        """
        :param tex:
        :param name:
        :return:
        """
        foo = open(tex + '/' + name, 'r').readlines()

        temp_ = []

        for sc in foo:
            temp = np.array(re.findall(pattern=r'\S[\d+\.\d+]*', string=sc))
            tep = [temp[0], temp[5], temp[6], temp[7], temp[10]]

            temp_.append(tep)

        result = np.array(temp_, dtype=np.int)

        snow2depth = np.array(result[:, -1], dtype=np.int)

        snow2depth_ = np.zeros((len(snow2depth)), dtype=np.int)

        for sc in range(len(snow2depth)):

            if (snow2depth[sc] >= 3) and (snow2depth[sc] <= 1000):
                snow2depth_[sc] = 1
            else:
                snow2depth_[sc] = 0

        result[:, -1] = snow2depth_

        return result


class ImageProcessing(TextProcessing, StationProcessing):
    """
    """
    def __init__(self, l1, l2, l3, l4, l5, l6, tex):
        """
        :param l1: BJ
        :param l2: DB
        :param l3: QT
        :param l4: Station
        :param l5: JAMS
        :param l6: RC
        :param tex: text
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5
        self.l6 = l6
        self.tex = tex

        super(ImageProcessing, self).__init__(l1=l1, l2=l2, l3=l3)
        self.index = self.look_up_table()
        self.sequence = self.is2seq_snow_season(tex=tex)
        self.station = self.station

    def cal_image_station(self):
        """
        :return:
        """
        sequence = np.array(self.get_seq_image()).flatten()
        print(sequence)
        """
        filename = [s.split('.')[0] for s in os.listdir(self.l5)]

        temp0, temp1, temp2 = [], [], []
        for name in tqdm.trange(len(filename)):

            if filename[name][0:6] in sequence:
                result_ = self.get_value(name=str(filename[name]))
                temp0.append(result_[0])
                temp1.append(result_[1])
                temp2.append(result_[2])

            else:
                continue

        result = {0: temp0, 1: temp1, 2: temp2}

        for kc in result.keys():
            if int(kc) == 0:

                content = result.get(kc)

                foo = open('./RESULT' + '/' + 'Image_BJ.txt', 'w')

                for cn in content:
                    for cm in cn:

                        if len(cm) == 0:
                            continue

                        else:
                            foo.writelines(str(cm[0]) + ','
                                           + str(cm[1]) + ','
                                           + str(cm[2]) + ','
                                           + str(cm[3]) + ',' + '\n')

                foo.close()

            if int(kc) == 1:

                content = result.get(kc)

                foo = open('./RESULT' + '/' + 'Image_DB.txt', 'w')

                for cn in content:
                    for cm in cn:

                        if len(cm) == 0:
                            continue

                        else:
                            foo.writelines(str(cm[0]) + ','
                                           + str(cm[1]) + ','
                                           + str(cm[2]) + ','
                                           + str(cm[3]) + ',' + '\n')
                foo.close()

            if int(kc) == 2:

                content = result.get(kc)

                foo = open('./RESULT' + '/' + 'Image_QT.txt', 'w')

                for cn in content:
                    for cm in cn:

                        if len(cm) == 0:
                            continue

                        else:
                            foo.writelines(str(cm[0]) + ','
                                           + str(cm[1]) + ','
                                           + str(cm[2]) + ','
                                           + str(cm[3]) + ',' + '\n')
                foo.close()
        """
    def get_value(self, name):
        """
        :param name:
        :return:
        """
        result = {}

        da2l5, da2l6 = self.get_data(l5=self.l5 + '/' + name + '.tif',
                                     l6=self.l6 + '/' + name + '_RC_L1.tif')

        for key in self.index.keys():

            temp = []
            content = self.index[key]
            for stn in content:

                temp.append([stn[0], name, da2l5[int(stn[1]), int(stn[2])], da2l6[int(stn[1]), int(stn[2])]])

            result[key] = temp

        return result

    def get_seq_image(self):
        """
        :return:
        """
        result = []

        sequence = [str(s).split('_')[6] for s in np.array(self.sequence).flatten()]
        filename = [str(s).split('.')[0][0:6] for s in np.array(os.listdir(self.l5))]

        for seq in range(0, len(sequence), 5):

            if sequence[seq] in filename:
                result.append(sequence[seq: seq + 5])

            else:
                continue

        return result
        
    @classmethod
    def get_data(cls, l5, l6):
        """
        :param l5:
        :param l6:
        :return:
        """
        return np.array(Rc.RasterTool.read_tiff(l5), dtype=np.int), np.array(Rc.RasterTool.read_tiff(l6), dtype=np.int)

    def look_up_table(self):
        """
        :param l4:
        :param l1:
        :param l2:
        :param l3:
        :return:
        """
        result = {}

        station = Rc.RasterTool.read_tiff(self.l4)

        stat = self.station
        for key in stat.keys():

            temp = []
            stat_ = stat[key]
            line, col = np.shape(station)

            for stn in range(len(stat_)):
                for ll in range(line):

                    if (station[ll, ::] == 9999).all():
                        continue

                    else:
                        for cc in range(col):

                            if np.int(station[ll, cc]) == np.int(stat_[stn]):
                                temp.append((stat_[stn], ll, cc))

                            else:
                                continue
            result[key] = temp

        return result


class TextResult(object):

    def cal_main(self, l1, l2, l3, lm1, lm2, lm3, csv2p):
        """
        :return:
        """
        xk, dk, tk = self.get_value(l1=l1, l2=l2, l3=l3, lm1=lm1, lm2=lm2, lm3=lm3, tag=True)
        xik, dik, tik = self.get_value(l1=l1, l2=l2, l3=l3, lm1=lm1, lm2=lm2, lm3=lm3, tag=False)

        select2xik = self.select_cloud2snow(data=xik)
        select2dik = self.select_cloud2snow(data=dik)
        select2tik = self.select_cloud2snow(data=tik)

        select2xk = self.opt_select_text_snow(data01=select2xik, data02=xk)
        select2dk = self.opt_select_text_snow(data01=select2dik, data02=dk)
        select2tk = self.opt_select_text_snow(data01=select2tik, data02=tk)

        sea2xk = self.cal_snow_season(df=select2xk)
        sea2dk = self.cal_snow_season(df=select2dk)
        sea2tk = self.cal_snow_season(df=select2tk)

        tex2xk = self.statical_value(df=sea2xk)
        print("======================================================")
        tex2dk = self.statical_value(df=sea2dk)
        print("======================================================")
        tex2tk = self.statical_value(df=sea2tk)

        df2xk = pd.DataFrame(tex2xk, columns=['season', 'a', 'b', 'c', 'd'])
        df2xk.to_csv(csv2p + '/' + 'XJ' + '.csv')

        df2dk = pd.DataFrame(tex2dk, columns=['season', 'a', 'b', 'c', 'd'])
        df2dk.to_csv(csv2p + '/' + 'DB' + '.csv')

        df2tk = pd.DataFrame(tex2tk, columns=['season', 'a', 'b', 'c', 'd'])
        df2tk.to_csv(csv2p + '/' + 'TQ' + '.csv')

    @classmethod
    def statical_value(cls, df):
        """
        :param df:
                 1             0  (image)
        ++++++++++++++++++++++++++++++++++
           1  |     a      |    b
        ++++++++++++++++++++++++++++++++++(text)
           0  |     c      |    d
        ++++++++++++++++++++++++++++++++++
        :return:
        """
        result = []
        for s in df.groupby(by=["flg"], axis=0):

            a, b, c, d = 0, 0, 0, 0
            x, y = np.array(s[1]["tex2snow"], dtype=np.int), np.array(s[1]["ml2snow"], dtype=np.int)

            date = np.array(s[1]["date"], dtype=np.str)
            season = date[0][0:4] + '-' + str(int(date[0][0:4]) + 1)

            for index in tqdm.trange(len(x)):

                if x[index] == 1 and y[index] == 1:
                    a += 1

                elif x[index] == 1 and y[index] == 0:
                    b += 1

                elif x[index] == 0 and y[index] == 1:
                    c += 1

                else:
                    d += 1

            result.append([season, a, b, c, d])

        return result

    @classmethod
    def cal_snow_season(cls, df):
        """
        :param df:
        :return:
        """

        df_ = pd.DataFrame(df)
        flg = np.array(np.sort(np.unique([str(s)[0:4] for s in df_['date']])), dtype=np.int)

        season = [[str(flg[s]) + '11',
                   str(flg[s]) + '12',
                   str(flg[s] + 1) + '01',
                   str(flg[s] + 1) + '02',
                   str(flg[s] + 1) + '03'] for s in range(len(flg) - 1)]

        tg = np.array([str(s)[0:6] for s in df_['date']], dtype=np.int)
        tg_ = np.zeros((len(tg)), dtype=np.int)

        for s in range(len(tg)):

            for c in range(len(season)):

                if str(tg[s]) in season[c]:
                    tg_[s] = c + 1

                else:
                    continue
        df_['flg'] = tg_

        return df_

    @classmethod
    def select_cloud2snow(cls, data):
        """
        :param data: [station, date, original_data(255), remove_cloud_data(250)[1 no snow, 2 snow]
        if original_date == cloud:
            save
        else:
            remove
        :return: 0 no snow, 1 snow
        """
        line, col = np.shape(data)

        result = []
        for xx in range(line):

            if (data[xx, 2] - 5) == data[xx, 3]:

                continue
            else:
                if data[xx, 3] == 2:
                    result.append([data[xx, 0], data[xx, 1], 1])
                else:
                    result.append([data[xx, 0], data[xx, 1], 0])

        return np.array(result, dtype=np.int)

    @classmethod
    def opt_select_text_snow(cls, data01, data02):
        """
        :param data01:  image station data
        :param data02:  observe data
        :return:
        """

        df2ml1 = pd.DataFrame(data01, columns=['station', 'date', 'ml2snow'])
        df2ml2 = pd.DataFrame(data02, columns=['station', 'date', 'tex2snow'])

        if len(df2ml1) <= len(df2ml2):

            df = pd.merge(left=df2ml1, right=df2ml2, how='inner', on=None, left_on=["station", 'date'],
                          right_on=["station", 'date'], left_index=False, right_index=False, sort=True,
                          suffixes=('_x', '_y'), copy=True, indicator=False,
                          validate="one_to_one")

            return df

        else:

            df = pd.merge(left=df2ml2, right=df2ml1, how='inner', on=None, left_on=["station", 'date'],
                          right_on=["station", 'date'], left_index=False, right_index=False, sort=True,
                          suffixes=('_x', '_y'), copy=True, indicator=False,
                          validate="one_to_one")

            return df

    @classmethod
    def get_value(cls, l1, l2, l3, lm1, lm2, lm3, tag=True):
        """
        :return:
        """
        if tag:
            foo_l1 = [[s.strip('\n').split(',')[0], s.strip('\n').split(',')[1], s.strip('\n').split(',')[2]] for s in
                      open(l1, 'r').readlines()]
            foo_l2 = [[s.strip('\n').split(',')[0], s.strip('\n').split(',')[1], s.strip('\n').split(',')[2]] for s in
                      open(l2, 'r').readlines()]
            foo_l3 = [[s.strip('\n').split(',')[0], s.strip('\n').split(',')[1], s.strip('\n').split(',')[2]] for s in
                      open(l3, 'r').readlines()]

            return np.array(foo_l1, dtype=np.int), np.array(foo_l2, dtype=np.int), np.array(foo_l3, dtype=np.int)

        else:

            foo_lm1 = [[s.strip('\n').split(',')[0], s.strip('\n').split(',')[1],
                        s.strip('\n').split(',')[2], s.strip('\n').split(',')[3]] for s in open(lm1, 'r').readlines()]
            foo_lm2 = [[s.strip('\n').split(',')[0], s.strip('\n').split(',')[1],
                        s.strip('\n').split(',')[2], s.strip('\n').split(',')[3]] for s in open(lm2, 'r').readlines()]
            foo_lm3 = [[s.strip('\n').split(',')[0], s.strip('\n').split(',')[1],
                        s.strip('\n').split(',')[2], s.strip('\n').split(',')[3]] for s in open(lm3, 'r').readlines()]

            return np.array(foo_lm1, dtype=np.int), np.array(foo_lm2, dtype=np.int), np.array(foo_lm3, dtype=np.int)


if __name__ == '__main__':
    OC_PATH = './RC_Data'
    RC_PATH = './RC_Data'
    STATION_PATH = './STAT/STAT.tif'

    # TextProcessing(l1='./STAT_CSV/STAT_BJ.csv',
    #                l2='./STAT_CSV/STAT_DB.csv',
    #                l3='./STAT_CSV/STAT_QT.csv').cal_observe_data(tex='./T_Data')
    #
    # TextProcessing(l1='./STAT_CSV/STAT_BJ.csv',
    #                l2='./STAT_CSV/STAT_DB.csv',
    #                l3='./STAT_CSV/STAT_QT.csv').optimize_cal(tex='./T_Data')
    # ImageProcessing(l1='./STAT_CSV/STAT_BJ.csv',
    #                 l2='./STAT_CSV/STAT_DB.csv',
    #                 l3='./STAT_CSV/STAT_QT.csv',
    #                 l4=STATION_PATH,
    #                 l5=JAMS_PATH,
    #                 l6=RC_PATH, tex='./T_Data').cal_image_station()
    TextResult().cal_main(l1='./RESULT/BJ.txt',
                          l2='./RESULT/DB.txt',
                          l3='./RESULT/QT.txt',
                          lm1='./RESULT/Image_BJ.txt',
                          lm2='./RESULT/Image_DB.txt',
                          lm3='./RESULT/Image_QT.txt',
                          csv2p='./RESULT')
