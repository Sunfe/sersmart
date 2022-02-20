# author:liao.zhicheng
# date:2021/07/16


import os
import sys
import time
import json
import yaml
import itertools
import re
import pandas as pd
import platform
import socket
import signal
import multiprocessing as mp
import threading
from itertools import groupby

from PyQt5.QtCore import (
         Qt,
         QCoreApplication,
         QObject,
         QSize,
         QRect,
         QThread,
         pyqtSignal,
         QMutex,
         QSemaphore,
         QEvent,
         QRegExp
    )

from PyQt5.QtGui import (
         QMouseEvent,
         QIntValidator,
         QRegExpValidator
    )


from PyQt5.QtWidgets import (
         QApplication,
         QWidget,
         QDesktopWidget,
         QMainWindow,
         QDialog,
         QDialogButtonBox,
         QMessageBox,
         QFileDialog,
         QLineEdit,
         QSizePolicy,
         QCheckBox
    )

from connection import *
from const import *
from logger import *

from ui_mainwindow import Ui_MainWindow
from ui_excelconvert import Ui_Dialog_excelconvert


def time_delay(delay):
    '''
    delay: second
    '''
    timedelta = 0
    time_start = time.time()
    while (timedelta < delay):
        cnt = 0
        while (cnt < 10000):
            cnt += 1
        time_end = time.time()
        timedelta = time_end - time_start
    return

def time_sleep(connection, time):
    '''
    time : second
    '''
    unit = 5
    if time < unit:
       time_delay(time)
       return
    else:
        cnt = 1
        while (cnt * unit < time):
            time_delay(unit)
            connection.write('\n')
            cnt += 1
        else:
            time_delay(time - (cnt - 1) * unit)
    return


mutex = QMutex()
class PrbsProcThread(QThread):
    MAX_PORT_PER_UNIT = 2

    def __init__(self, config=None, login=None, port=0, queue=None):
        super().__init__()

        self.login = login
        self.config = config
        self.slot = login.slot
        self.port = port
        self.queue = queue

        self.finished_records = None
        self.total_cnt = self.calc_total_round_cnt(config)
        self.left_cnt = 0
        self.semaphore = QSemaphore(1)
        self.logger = Logger("sersmart_log_slot{}.txt".format(self.slot)).logger

    @property
    def name(self):
        return self.objectName()

    def run(self):
        self.start_prbs(self.port)

    def update_login(self, login):
        '''
        self.login = login
        '''
        pass

    def update_connection(self):
        '''
        self.con = self.login.con
        '''
        pass

    def exe_commands(self, commands, sleep):
        for command in commands:
            logger.info("{}: {}\n".format(self.name, command))
            self.login.write("{}\n".format(command))
            time_delay(sleep)
            print("{}:\n{}".format(self.name, self.login.read(method=c.READ_VERY_EAGER)))
        return

    def port2unit(self, port=0):
        '''
        编排规则：
        一个unit共2个port， 每个port 4根serdes，serdes编号范围：port0：0 - 3， port1：4-7
        c600共1个unit, port取值: 0 - 1
        c89e共3个unit, port取值：0 - 5
        '''
        unit = port // self.MAX_PORT_PER_UNIT
        return unit

    def calc_total_round_cnt(self, config):
        '''
        计算总共需要循环的次数
        '''
        target = config['target']
        opt_tx = config['opt_tx']
        tx_ce0 = config['ffe']['txce0']
        tx_ce1 = config['ffe']['txce1']
        tx_ce2 = config['ffe']['txce2']

        total_cnt = 0
        ports = target[self.slot]
        for i in itertools.product(ports, opt_tx, tx_ce0, tx_ce1, tx_ce2):
            total_cnt = total_cnt + 1
        return total_cnt

    def init_global_param(self, xfwdp, port):
        print("{}: begin to init global praram.".format(self.name))
        unit = self.port2unit(port)
        cmd = ["gal_debug_disable(7)",
               "diagMsSerdesSetDetectControlOff()",
               "fpp_lif_adeq_stop({}, 0xff)".format(unit)]
        self.exe_commands(cmd, 0.5)

        # 换页操作，将光模块寄存器PAGE0 BYTE127写0x03，换页到PAGE3
        print("{}: begin to init page.".format(self.name))
        cmd = ["data=0x03",
               "BSP_CHIP_OptI2cWrite(1, 0, 127, &data, 1)",
               "diagbspchipOptI2cRead(1, 0, 127)"]
        self.exe_commands(cmd, 0.5)

        cmd = ["fpp_serdes_hrst_set({}, 1, 1)".format(unit),
               "fpp_serdes_hrst_set({}, 1, 0)".format(unit),
               "fpp_serdes_single_init({}, 30, 6)".format(unit)]
        self.exe_commands(cmd, 0.5)

        if xfwdp:
            print("{}: begin to reverse polarity, xfwdp only.".format(self.name))
            self.reverse_polarity(port)

    def reverse_polarity(self, port):
        '''
        极性反转,仅XFWDP需要执行
        '''
        unit = self.port2unit(port)
        cmd = [
            "fpp_serdes_tx_data_inv_en_set({}, 0, 0)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 1, 0)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 2, 0)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 3, 1)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 4, 1)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 5, 0)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 6, 0)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 7, 0)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 0, 1)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 1, 1)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 2, 1)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 3, 0)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 4, 1)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 5, 1)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 6, 1)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 7, 0)".format(unit)
        ]
        self.exe_commands(cmd, 0.5)
        return

    def optical_equalize(self, opt_tx):
        '''
        设置光模块均衡
        '''
        print("{}: begin to set optical module equalization.".format(self.name))
        cmd = [
            "data={}".format(opt_tx),
            "BSP_CHIP_OptI2cWrite(1, 0, 234, &data, 1)",
            "diagbspchipOptI2cRead(1, 0, 234)",
            "BSP_CHIP_OptI2cWrite(1, 0, 235, &data, 1)",
            "diagbspchipOptI2cRead(1, 0, 235)"
        ]
        self.exe_commands(cmd, 0.5)
        return

    def init_serdes_ffe(self, port, ce0, ce1, ce2, serdes):
        '''
        设置FFE参数
        '''
        unit = self.port2unit(port)
        print("{}: begin to set FFE parameters.".format(self.name))
        for serdes_id in serdes:
            self.login.write("fpp_serdes_ffe_set_test({}, {}, {}, {}, {})\n".format(unit, serdes_id, ce0, ce1, ce2))
            time_delay(0.5)
            self.logger.info("{}: {}".format(self.name, self.login.read_buf()))

    def clear_previous_prbs_error_flag(self, port, serdes):
        #先发送一次PRBS
        print("{}: begin to clear previous prbs error flag".format(self.name))
        print("{}: firstly, send prbs signal".format(self.name))

        unit = self.port2unit(port)
        for serdes_id in serdes:
            self.login.write("fpp_serdes_prbs_gen_en_set({}, {}, 7, 1)\n".format(unit, serdes_id))
            time_delay(0.5)
            self.logger.info(self.login.read_buf())

        # 等待2s
        print("{}: waitfor 2s...".format(self.name))
        time_delay(2)

        # 清除误码, 该组命令每1s执行一次, 若干次, 确保无误码
        clr_ef_cnt = self.config['clear_errorflag_count']
        print("{}: secondly, read errorflag, operates {} times.".format(self.name, clr_ef_cnt))
        for i in range(clr_ef_cnt):
            print("{}: {} times".format(self.name, i + 1))
            for serdes_id in serdes:
                self.login.write("fpp_serdes_prbs_chk_en_set({}, {}, 7, 1)\n".format(unit, serdes_id))
                time_delay(0.5)
                self.logger.info(self.login.read_buf())

    def generate_one_record(self, port, opt_tx, tx_ce0, tx_ce1, tx_ce2, serdes_error):
        row = {'port': port, 'opt_tx': opt_tx, "tx_ce0": tx_ce0, "tx_ce1": tx_ce1, "tx_ce2": tx_ce2}
        if '1' in serdes_error:
            row['error'] = 1
        elif 'E' in serdes_error:
            row['error'] = -1
        else:
            row['error'] = 0
        return row

    def start_prbs(self, port):
        xfwdp  = self.config['xfwdp']
        opt_tx = self.config['opt_tx']
        tx_ce0 = self.config['ffe']['txce0']
        tx_ce1 = self.config['ffe']['txce1']
        tx_ce2 = self.config['ffe']['txce2']

        prbs_df = pd.DataFrame(columns=['port', 'opt_tx', 'tx_ce0', 'tx_ce1', 'tx_ce2', 'error'])
        self.finished_records = self.read_finished_records(port)

        print("{}: begin to start prbs, port {}...".format(self.name, port))
        print("{}: there will be {} times to loop for this slot".format(self.name, self.total_cnt))

        serdes_range = list(range(0,8))
        if port % self.MAX_PORT_PER_UNIT == 0:
            serdes = serdes_range[0:4]
        else:
            serdes = serdes_range[4:8]

        mutex.lock()
        print("{}: ++++++++++ mutex locked! ++++++++++".format(self.name))
        self.init_global_param(xfwdp, port)
        mutex.unlock()
        print("{}: ++++++++++ mutex released! ++++++++++".format(self.name))

        cnt = 0
        for opt in opt_tx:
            for ce0, ce1, ce2 in itertools.product(tx_ce0, tx_ce1, tx_ce2):
                if self.finished_records is not None:
                    condition = "port=={} & opt_tx=={} & tx_ce0=={} & tx_ce1=={} & tx_ce2=={}"\
                                .format(port, opt, ce0, ce1, ce2)
                    sel = self.finished_records.query(condition)
                    if self.finished_records.shape[0] > 0 and sel.shape[0] > 0:
                        prbs_df = prbs_df.append(sel, ignore_index=True)
                        print(
                                "{}: slot:{} port:{} opt_tx:{} tx_ce0:{} tx_ce1:{} tx_ce2:{}, "\
                                "already finished, bypass!"\
                                .format(self.name, self.slot, port, opt, ce0, ce1, ce2)
                        )
                        index = sel.index.tolist()
                        self.finished_records.drop(index, inplace=True)

                        cnt = cnt + 1
                        self.left_cnt = self.total_cnt - cnt

                        continue
                    else:
                        row = self.exec_prbs(port, opt, ce0, ce1, ce2, serdes)
                else:
                    row = self.exec_prbs(port, opt, ce0, ce1, ce2, serdes)

                print("{}: to append row to dataframe".format(self.name))
                prbs_df = prbs_df.append(pd.Series(row), ignore_index=True)
                print("{}: done!".format(self.name))
                print("{}: slot{}: current dataframe:\n{}".format(self.name, self.slot, prbs_df))

                # 每次测试完成后都进行保存, 避免测试结果中途丢失
                self.export(port, prbs_df)

                cnt = cnt + 1
                self.left_cnt = self.total_cnt - cnt
                print("{}: slot {} port {}: loop {} complete, there should be {} left!\n"
                      .format(self.name, self.slot, port, cnt, self.left_cnt))

        print("{}:\n----------prbs 100G loop result----------".format(self.name))
        print("{}:\n{}".format(self.name, prbs_df))
        print("{}: port {}: all parameters looped, total count {}, test complete!".format(self.name, port, cnt))

        return prbs_df

    def exec_prbs(self, port, opt, ce0, ce1, ce2, serdes):
        print("{}: start a new loop, port:{},opt:{},ce0:{},ce1:{},ce2{}"\
              .format(self.name, port, opt, ce0, ce1, ce2))
        serdes_error = self.exec_prbs_start(port, opt, ce0, ce1, ce2, serdes)
        print("{}: serdes errorflag: {}".format(self.name, serdes_error))

        print("{}: to generate one record".format(self.name))
        row = self.generate_one_record(port, opt, ce0, ce1, ce2, serdes_error)
        print("{}: row:{}".format(self.name, row))

        print("{}: to stop prbs test".format(self.name))
        self.exec_prbs_stop(port, serdes)
        print("{}: done!".format(self.name))

        return row

    def exec_prbs_start(self, port, opt, ce0, ce1, ce2, serdes):
        mutex.lock()
        print("{}: ++++++++++ mutex locked! ++++++++++".format(self.name))
        self.init_serdes_ffe(port, ce0, ce1, ce2, serdes)
        self.clear_previous_prbs_error_flag(port, serdes)
        mutex.unlock()
        print("{}: ++++++++++ mutex released! ++++++++++".format(self.name))

        # 静默一个时间段, 然后进行PRBS测试, 并检查结果
        wait_time  = self.config['wait_time']
        print("{}: waitfor {}s...".format(self.name, wait_time))
        time_delay(wait_time)

        mutex.lock()
        print("{}: ++++++++++ mutex locked! ++++++++++".format(self.name))
        print(
             "{}: begin to start prbs:"\
             "slot{}, port {}, opt {}, ce0 {}, ce1 {}, ce2 {}"\
             .format(self.name, self.slot, port, opt, ce0, ce1, ce2)
        )

        self.login.clearbuf()

        serdes_error=[]
        recover_cnt = 0
        id_index = 0
        unit = self.port2unit(port)
        while id_index < len(serdes):
            try:
                self.login.write("fpp_serdes_prbs_chk_en_set({}, {}, 7, 1)\n"
                               .format(unit, serdes[id_index]))
                time_delay(0.5)
                buf = self.login.read_buf()
                print("{}: {}".format(self.name, buf))

                try_cnt = 0
                while buf.find('Pattern checker errorflag') < 0:
                   time_delay(0.5)
                   buf = self.login.read_buf()
                   try_cnt += 1
                   if try_cnt > 10:
                       logger.info("{}: Pattern checker errorflag not found, tried {} times"\
                                   .format(self.name, try_cnt))
                       raise Exception("fpp_serdes_prbs_chk_en_set not respond correctly!")
                       break

                res_str = buf.splitlines(False)
                # res_str[0]: 命令本身, res_str[1]:errorflag, rest_str[2]:error_number
                error_code = re.findall(r'Pattern checker errorflag         :0x([0-1])', res_str[1])
                serdes_error.append(error_code[0])
            except Exception as e:
                    self.logger.error("{}: {}".format(self.name, repr(e)))
                    self.login.reconnect()
                    #self.update_connection()

                    # 清空缓存, 不然内容残留,会导致出错上面的buf.splitlines多出一些内容,
                    # 导致解析出错
                    self.login.clearbuf()
                    # 网络未知异常, 放在这里处理, 避免一次出现问题导致后续
                    # 终止的情况, 尝试2次后还不成功, 结果统一填写E
                    if recover_cnt < 2:
                        id_index -= 1
                        recover_cnt += 1
                    else:
                        self.logger.error(str(e))
                        serdes_error.append("E")
            finally:
                   id_index += 1

        mutex.unlock()
        print("{}: ++++++++++ mutex released! ++++++++++".format(self.name))

        return serdes_error

    def exec_prbs_stop(self, port, serdes):
        unit = self.port2unit(port)
        for serdes_id in serdes:
            self.login.write("fpp_serdes_prbs_gen_en_set({}, {}, 7, 0)\n".format(unit, serdes_id))
            time_delay(0.5)
            buf = self.login.read_buf()
            print(buf)
        self.login.clearbuf()
        return

    def read_finished_records(self, port):
        file_orig = "result_orig_{}_{}.xlsx".format(self.slot, port+1)
        file_tile = "result_tile_{}_{}.xlsx".format(self.slot, port+1)
        file_split = "result_split_{}_{}.xlsx".format(self.slot, port+1)
        export = Export(file_orig, file_tile, file_split)
        return export.from_excel()

    def export(self, port, result):
        file_orig = "result_orig_{}_{}.xlsx".format(self.slot, port+1)
        file_tile = "result_tile_{}_{}.xlsx".format(self.slot, port+1)
        file_split = "result_split_{}_{}.xlsx".format(self.slot, port+1)
        export = Export(file_orig, file_tile, file_split)
        export.to_excel(result)


class Login(Telnet):
    def __init__(self, system, ip, port, slot):
        super(__class__, self).__init__(ip, port)

        self.system = system
        self.ip = ip
        self.port = port
        self.slot = slot

    def update_connection(connection):
        #self.con = connection
        pass

    def login_board_c600(self):
        ip = "168.1.{}.0 10000".format(129 + self.slot)
        error_str = "login slot {} error".format(self.slot)

        self.write('\n\n')
        time_delay(1)
        buf = self.read_buf()
        if buf.find("/ #") > 0:
            # 已经在主控shell下
            pass
        elif buf.find('\[FTMETHLP\]#'):
            # 如果在主控shell下，直接telnet到线卡
            # 如果线卡shell下,先exit当前线卡,再telnet到其他线卡
            self.write("exit\n")
        else:
            # 既不在主控shell，也不在线卡shell([FTMETHLP])
            raise Exception(error_str)

        self.write('\n\n')
        if self.waitfor('/ #', 2) == False:
            raise Exception(error_str)
        self.write("telnet {}\n".format(ip))

        if self.waitfor('login:', 2) == False:
            raise Exception(error_str)
        self.write("zxos\n")

        if self.waitfor('password:', 2) == False:
            raise Exception(error_str)
        self.write("zxos{}fnscp@$%\n".format(self.slot))

        if self.waitfor('Successfully login into ushell', 2) == False:
            raise Exception(error_str)
        self.write("\n")

        if self.waitfor('\[admin\]', 2) == False:
            raise Exception(error_str)
        self.write("shell ftm\n")

        if self.waitfor('Now switch to FTMETHLP shell', 2) == False:
            raise Exception(error_str)
        self.write("\n")

        if self.waitfor('\[FTMETHLP\]#', 2) == False:
            raise Exception(error_str)
        self.write("\n")
        return

    def login_board_c89e(self):
        ip = "168.0.{}.1 10000".format(129 + self.slot)
        error_str = "loggin slot {} error".format(self.slot)

        self.write('\n\n')
        time_delay(1)
        buf = self.read_buf()
        if buf.find("/ #") > 0:
            # 已经在主控shell下
            pass
        elif buf.find('\[FTMETHLP\]#'):
            # 如果在主控shell下，直接telnet到线卡
            # 如果线卡shell下,先exit当前线卡,再telnet到其他线卡
            self.write("exit\n")
        else:
            # 既不在主控shell，也不在线卡shell([FTMETHLP])
            raise Exception(error_str)

        self.write('\n\n')
        if self.waitfor('/ #', 2) == False:
            raise Exception(error_str)
        self.write("telnet {}\n".format(ip))

        if self.waitfor('login:', 2) == False:
            raise Exception(error_str)
        self.write("zte\n")

        if self.waitfor('password:', 2) == False:
            raise Exception(error_str)
        self.write("zte\n")

        if self.waitfor('Successfully login into ushell', 2) == False:
            raise Exception(error_str)
        self.write("\n\n\n")

        if self.waitfor('\[admin\]', 2) == False:
            raise Exception(error_str)
        self.write("shell fcm\n")

        if self.waitfor('Now switch to FCMMGRPF_F shell', 2) == False:
            raise Exception(error_str)
        self.write("\n")

        if self.waitfor('\[FCMMGRPF_F\]#', 2) == False:
            raise Exception(error_str)
        self.write("\n")
        return

    def connect_board(self):
        if self.system == 'c600':
            self.login_board_c600()
        elif self.system == 'c89e':
            self.login_board_c89e()
        return

    def reconnect(self):
        self.close()
        logger.info("reconnect slot {}".format(self.slot))

        cnt = 0
        total_try_times = 10000
        while cnt < total_try_times:
            try:
                self.open(self.ip, self.port)
                logger.info("{} times has been tried, succeed".format(cnt))
                break
            except Exception as e:
                logger.error("reconnect {} times, failed!".format(cnt))
                logger.error(str(e))
                logger.error(repr(e))

                cnt += 1
                time_delay(10)
        else:
            raise Exception("NetworkBlock")
            return

        self.connect_board()
        return


class Export():
    def __init__(self, file_orig, file_tile, file_split):
        self.file_orig = file_orig
        self.file_tile = file_tile
        self.file_split = file_split

    def to_excel(self, result):
        if len(result) == 0 :
            logger.error("no data in result dataframe!")
            return

        logger.info("begin to export to xlsx....")
        try:
            writer = pd.ExcelWriter(self.file_orig, engine='xlsxwriter')
            result.to_excel(writer, sheet_name='Sheet1', index=False)

            workbook = writer.book
            worksheet = writer.sheets['Sheet1']

            # 表头上色
            row_header_format = workbook.add_format({
                'bg_color': '#CEE3F6',
                'bold':  True,
                'border': 1
                }
            )
            for col_num, value in enumerate(result.columns.values):
                worksheet.write(0, col_num, value, row_header_format)

            # 错误标识上色
            error_format = workbook.add_format({
                'bg_color': 'red',
                'font_color': 'black'})
            worksheet.conditional_format(1, 5, len(result), 5, {
                'type':     'text',
                'criteria': 'containing',
                'value':    '1',
                'format':    error_format
                }
            )

            writer.save()
            writer.close()
            logger.info("done!")
        except Exception as e:
            logger.info("failed, error:{}".format(str(e)))
        return

    def to_excel_tile(self, result):
        if len(result) == 0 :
            logger.error("no data in result dataframe!")
            return

        logger.info("begin to export split result to xlsx....")
        result.set_index(['port','opt_tx','tx_ce0', 'tx_ce1', 'tx_ce2'], inplace=True)
        result = result.unstack(fill_value=0)
        result[result == 1] = 'F'

        # 多级索引平铺为一张表格
        row_margin, col_margin, row_offset = [5, 5, 1]
        with pd.ExcelWriter(self.file_tile) as writer:
            result_tile = result.style.set_properties(**{'text-align': 'center'})
            result_tile.to_excel(writer)

            workbook = writer.book
            worksheet = writer.sheets['Sheet1']

            row_num, col_num = result.shape
            error_format = workbook.add_format({
                'bg_color': 'red',
                'font_color': 'black'})
            worksheet.conditional_format(*[0, 0, row_num + row_margin, 20], {
                'type':     'text',
                'criteria': 'containing',
                'value':    'F',
                'format':    error_format})

            worksheet.set_column(4, 14, 6)

            writer.save()
            writer.close()

        # 解多级索引, 拆封为多张表格
        # 取得前3级引用的value
        row_index_port = result.index.get_level_values(0).drop_duplicates()
        row_index_opt_tx = result.index.get_level_values(1).drop_duplicates()
        row_index_tx_ce0 = result.index.get_level_values(2).drop_duplicates()

        with pd.ExcelWriter(self.file_split) as writer:
            for i, j, k in itertools.product(row_index_port, row_index_opt_tx, row_index_tx_ce0):
                # 解多级索引, 留下tx_ce1, txce2
                df_txce_1_0 = result.loc[(i, j, k)]

                row_num, col_num = df_txce_1_0.shape
                df_txce_1_0 = df_txce_1_0.style.set_properties(**{
                    'text-align':  'center',
                    'font-family': 'Times New Roman'})
                df_txce_1_0.to_excel(writer, sheet_name='Sheet1', startrow=row_offset, startcol=0)

                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                error_format = workbook.add_format({
                    'bg_color': 'red',
                    'font_color': 'black'})

                first_row, first_col = [row_offset, 0]
                last_row, last_col = [row_offset + row_num + row_margin,  col_num + col_margin]
                worksheet.conditional_format(first_row, first_col, last_row, last_col, {
                    'type':     'text',
                    'criteria': 'containing',
                    'value':    'F',
                    'format':    error_format})

                worksheet.set_column(0, 0, 10)
                worksheet.set_column(1, col_num + 1, 6)

                comment_row, comment_col = [row_offset - 1, 0]
                cell_format = workbook.add_format({
                    'bold':       True,
                    'font_color': 'black',
                    'bg_color':   '#B8CCE4',
                    'font_name':  'Times New Roman'})

                for m in range(col_num + 1):
                    worksheet.write(comment_row, m, None, cell_format)
                worksheet.write(comment_row, comment_col,
                     "port:{}, opt_tx:{}, tx_ce0:{}".format(i, j, k),
                     cell_format)

                row_offset += row_num + row_margin

            writer.save()
            writer.close()
            logger.info("done!")
            return

    def from_excel(self):
        try:
            data = pd.read_excel(self.file_orig, sheet_name="Sheet1")
        except Exception as e:
            logger.error(str(e))
            logger.error(repr(e))
            data = None
        finally:
            return data


def exit(signum, frame):
    logg.info('sersmart terminated by user\n')
    os._exit(0)
    return


class MsgDispathThread(QThread):
    signal = pyqtSignal(str)

    def __init__(self, queue, parent=None):
        QThread.__init__(self, parent)
        self.queue = queue

    def run(self):
        while True:
            if self.queue.empty():
                continue

            text = self.queue.get(True)
            self.signal.emit(text)
        return


class Sersmart(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Sersmart, self).__init__(parent)

        self.config = None
        self.login = []
        self.threads = []

        self.lineEdit_target = {}
        self.setupUi(self)

        self.setFixedSize(self.width(), self.height())

        self.slotValidator = QRegExpValidator(self)
        self.slotValidator.setRegExp(QRegExp('([1-9]|1[0-9]|20)'))
        self.lineEdit_slot_0.setValidator(self.slotValidator)

        self.portValidator = QRegExpValidator(self)
        self.portValidator.setRegExp(
            QRegExp(
               '(([1-5]-[2-6],)|([1-6],)){,6}(([1-5]-[2-6],)|([1-6],)){,6}'
            )
        )
        self.lineEdit_port_0.setValidator(self.portValidator)

        self.queue = mp.Queue()
        self.msg_receive = MsgDispathThread(self.queue)
        self.msg_receive.signal.connect(self.set_browse_text)
        self.msg_receive.start()

        self.pushButton_startprbs.clicked.connect(self.on_btnclick_start_prbs)
        self.pushButton_stopprbs.clicked.connect(self.on_btnclick_stop_prbs)
        self.pushButton_addtarget.clicked.connect(self.add_target)
        self.pushButton_deltarget.clicked.connect(self.del_target)

        self.init_action()

    def init_action(self):
        self.openFileAction.triggered.connect(self.open_file)
        self.exitAction.triggered.connect(self.exit_app)
        self.actionExcelConvertor.triggered.connect(self.convert_excel)


    def init_config():
        cur_path = os.path.abspath(".")
        config_file = os.path.join(cur_path, c.CONFIG_FILE)

        if platform.system() == 'Windows':
            config_file = '\\\\'.join(config_file.split('\\'))

        try:
            with open(config_file, mode='r',encoding="UTF-8") as f:
                config = yaml.load(f.read())
        except Exception as e:
             logger.info("{} cannot be opened, reject to continue!".format(c.CONFIG_FILE))
             os._exit(0)

        finally:
            return config

    def check_config(config):
        for port in config['port']:
            if port not in range(0, 6):
                logger.error("error portid config!")
                os._exit(0)
        return


    def set_browse_text(self, text):
        self.textBrowser_output.append(text)
        # 实时刷新
        #QApplication.processEvents()


    def open_file(self):
        fname = QFileDialog.getOpenFileName(self, '打开新文件',"","*.yaml")
        #fname[0]:实际文件名,包含路径, fname[1]:文件后缀名
        if fname[0] == '':
            return

        try:
            with open(fname[0], mode='r',encoding="UTF-8") as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
                self.comboBox_system.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.critical(self, "异常", str(e), QMessageBox.Ok)
            return

        if self.config is None:
            return

        index = 0 if self.config['system'] == 'c600' else 1
        self.comboBox_system.setCurrentIndex(index)

        self.lineEdit_ip.setText(self.config['ip'])

        state = Qt.Checked if self.config['xfwdp'] == True else Qt.Unchecked
        self.checkBox_xfwdp.setCheckState(state)

        text = str(self.config['opt_tx']).strip('[').strip(']')
        self.lineEdit_opt_tx.setText(text)

        text = self.pack_numbers(self.config['ffe']['txce0'])
        self.lineEdit_txce0.setText(text)

        text = self.pack_numbers(self.config['ffe']['txce1'])
        self.lineEdit_txce1.setText(text)

        text = self.pack_numbers(self.config['ffe']['txce2'])
        self.lineEdit_txce2.setText(text)

        target = self.config['target']
        slots = list(target.keys())
        # 增加slot, port文本框, 第一行文本框已经在UI中静态添加
        for i in range(1, len(slots)):
            self.add_target()

        # 写入配置
        a = self.lineEdit_target
        b = self.gridLayout_target
        for i in range(0, len(slots)):
            slot = slots[i]
            if i==0:
               self.lineEdit_slot_0.setText(str(slot))
               self.lineEdit_port_0.setText(str(target[slot]).strip('[').strip(']'))
            else:
               a['{}-{}'.format(i+1, 0)].setText(str(slot))
               # ports:
               a['{}-{}'.format(i+1, 1)].setText(str(target[slot]).strip('[').strip(']'))

        text = str(self.config['wait_time'])
        self.lineEdit_wait_time.setText(text)
        text = str(self.config['clear_errorflag_count'])
        self.lineEdit_clr_cnt.setText(text)

        return

    def pack_numbers(self, a):
        fun = lambda x: x[1]-x[0]
        for k, g in groupby(enumerate(a), fun):
            l1 = [j for i, j in g]
            if len(l1) > 1:
                scop = str(min(l1)) + '-' + str(max(l1))
            else:
                scop = l1[0]
        return scop

    def expand_packed_range(self, scope):
        num = []
        first_layer = scope.split(',')
        for numstr in first_layer:
            if '-' not in numstr:
                num.append(int(numstr))
                continue

            second_layer = numstr.split('-')
            if len(second_layer) > 2    :
                QMessageBox.critical(self, "error", "检查配置", QMessageBox.Ok)
                return 0xffff

            if len(second_layer) == 2:
                # 比如:'1-15'转换为1, 2, ...
                i = second_layer[0]
                j = second_layer[1]
                for n in range(int(i), int(j) + 1):
                    num.append(n)
            else:
                # 比如:'1'转换为1
                num.append(int(second_layer[0]))

        return num


    def add_target(self):
        a = self.lineEdit_target
        b = self.gridLayout_target
        bottom_margin = 5
        b.setContentsMargins(0, 0, 0, bottom_margin)

        row = b.rowCount()
        if (row > 3):
            QMessageBox.critical(self, "warning", "超过最大支持数", QMessageBox.Ok)
            return

        w_slot = self.lineEdit_slot_0.width()
        h_slot = self.lineEdit_slot_0.height()
        height = h_slot + bottom_margin

        # 现有控件下移
        pos_x = self.pushButton_addtarget.geometry().x()
        pos_y = self.pushButton_addtarget.geometry().y()
        self.pushButton_addtarget.move(pos_x , pos_y + height)

        pos_x = self.pushButton_deltarget.geometry().x()
        pos_y = self.pushButton_deltarget.geometry().y()
        self.pushButton_deltarget.move(pos_x , pos_y + height)

        pos_x = self.checkBox_xfwdp.geometry().x()
        pos_y = self.checkBox_xfwdp.geometry().y()
        self.checkBox_xfwdp.move(pos_x , pos_y + height)

        # 创建和第一行尺寸一样的控件,分别放在0, 1, 2三列
        edit_slot = QLineEdit(self.gridLayoutWidget)
        edit_slot.setFixedSize(w_slot, h_slot)
        edit_slot.setValidator(self.slotValidator)
        a["{}-{}".format(row, 0)] = edit_slot

        w_port = self.lineEdit_port_0.width()
        h_port = self.lineEdit_port_0.height()
        edit_port = QLineEdit(self.gridLayoutWidget)
        edit_port.setFixedSize(w_port, h_port)
        edit_port.setValidator(self.portValidator)
        a["{}-{}".format(row, 1)] = edit_port

        w_port = self.checkBox_select_0.width()
        h_port = self.checkBox_select_0.height()
        checkbox_select = QCheckBox(self.gridLayoutWidget)
        checkbox_select.setFixedSize(w_port, h_port)
        a["{}-{}".format(row, 2)] = checkbox_select

        # 将创建的控件加入到gridLayout
        b.addWidget(a["{}-{}".format(row, 0)], row, 0, 1, 1)
        b.addWidget(a["{}-{}".format(row, 1)], row, 1, 1, 1)
        b.addWidget(a["{}-{}".format(row, 2)], row, 2, 1, 1)

        return

    def del_target(self):
        row_cnt = self.gridLayout_target.rowCount()
        print(row_cnt)
        for i in range(2, row_cnt):
            a = self.lineEdit_target
            if a["{}-{}".format(i, 2)].isChecked():
                print("{} is checked".format(i))
                self.gridLayout_target.removeWidget(a["{}-{}".format(i, 0)])
                self.gridLayout_target.removeWidget(a["{}-{}".format(i, 1)])
                self.gridLayout_target.removeWidget(a["{}-{}".format(i, 2)])
            else:
                pass

        return

    def get_config_from_ui(self):
        config = {}

        config['system'] = self.comboBox_system.currentText()
        config['ip'] = self.lineEdit_ip.text()
        config['port'] = 23

        if Qt.Checked == self.checkBox_xfwdp.checkState():
            config['xfwdp'] = True
        else:
            config['xfwdp'] = False

        # slot-port QGridLayout 第一行
        c = config['target'] = {}
        slot = int(self.lineEdit_slot_0.text())
        port_str = self.lineEdit_port_0.text()
        c[slot] = self.expand_packed_range(port_str)

        # slot-port QGridLayout 第一行之后的其他行
        for row in range(2, self.gridLayout_target.rowCount()):
            slot = int(self.lineEdit_target['{}-{}'.format(row, 0)].text())
            port_str = self.lineEdit_target['{}-{}'.format(row, 1)].text()
            c[slot] = self.expand_packed_range(port_str)

        config['opt_tx'] = self.expand_packed_range(self.lineEdit_opt_tx.text())

        ffe = config['ffe'] = {}
        ffe['txce0'] = self.expand_packed_range(self.lineEdit_txce0.text())
        ffe['txce1'] = self.expand_packed_range(self.lineEdit_txce1.text())
        ffe['txce2'] = self.expand_packed_range(self.lineEdit_txce2.text())
        config['wait_time'] = int(self.lineEdit_wait_time.text())
        config['clear_errorflag_count'] = int(self.lineEdit_clr_cnt.text())
        self.config = config

        return config


    def on_btnclick_start_prbs(self):
        if len(self.lineEdit_ip.text()) == 0:
            QMessageBox.critical(self, "错误", "ip不能为空!", QMessageBox.Ok)
            self.lineEdit_ip.setFocus()
            return

        if len(self.lineEdit_slot_0.text()) == 0:
            QMessageBox.critical(self, "错误", "槽位号不能为空!", QMessageBox.Ok)
            self.lineEdit_slot_0.setFocus()
            return


        if len(self.lineEdit_port_0.text()) == 0:
            QMessageBox.critical(self, "错误", "端口号不能为空!", QMessageBox.Ok)
            self.lineEdit_port_0.setFocus()
            return

        config = self.get_config_from_ui()
        if not bool(config):
            QMessageBox.critical(self, "错误", "配置未加载!", QMessageBox.Ok)
            return

        if len(self.lineEdit_opt_tx.text()) == 0:
            QMessageBox.critical(self, "错误", "opt_tx参数不能为空!", QMessageBox.Ok)
            self.lineEdit_opt_tx.setFocus()
            return

        if len(self.lineEdit_txce0.text()) == 0:
            QMessageBox.critical(self, "错误", "tx_ce0参数不能为空!", QMessageBox.Ok)
            self.lineEdit_txce0.setFocus()
            return

        if len(self.lineEdit_txce1.text()) == 0:
            QMessageBox.critical(self, "错误", "tx_ce1参数不能为空!", QMessageBox.Ok)
            self.lineEdit_txce1.setFocus()
            return

        if len(self.lineEdit_txce2.text()) == 0:
            QMessageBox.critical(self, "错误", "tx_ce2参数不能为空!", QMessageBox.Ok)
            self.lineEdit_txce2.setFocus()
            return


        # 按钮变灰
        self.pushButton_startprbs.setEnabled(False)
        self.pushButton_startprbs.setEnabled(True)

        # 每个槽位一个Login, 每个端口一个线程
        target = config['target']
        PRO_NUM = len(target)
        for slot in target.keys():
            # 初始化连接, 每个进程需要一个连接
            login = Login(
                config['system'],
                config['ip'],
                config['port'],
                slot
            )
            self.login.append(login)

            msg = "start to login slot {}".format(slot)
            logger.info(msg)
            self.queue.put(msg)

            try:
                login.connect_board()
                logger.info("successfully!")
            except Exception as e:
                logger.error("failed!")
                logger.error(str(e))
                continue

            for port in target[slot]:
                thread = PrbsProcThread(config, login, port, self.queue, )
                thread.setObjectName("task_{}_{}".format(slot, port))
                self.threads.append(thread)

            for thread in self.threads:
                thread.start()
                time_delay(30)

    def on_btnclick_stop_prbs(self):
        # 开始按钮变亮, 停止按钮变灰
        self.pushButton_startprbs.setEnabled(True)
        self.pushButton_stopprbs.setEnabled(False)

        return

    def convert_excel(self):
        dialog = Excelconvert()
        dialog.lineEdit_src.setFocus()
        dialog.exec_()

    def exit_app(self):
        QCoreApplication.instance().quit()
        return


class Excelconvert(QDialog, Ui_Dialog_excelconvert):
    def __init__(self, parent=None):
        super(Excelconvert, self).__init__(parent)
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())

        self.pushButton_src.clicked.connect(self.on_btnclick_src)
        self.pushButton_dst.clicked.connect(self.on_btnclick_dst)

        self.buttonBox.button(QDialogButtonBox.Ok).installEventFilter(self)
        self.buttonBox.accepted.connect(self.on_accepted)
        self.buttonBox.rejected.connect(self.on_rejected)

    def eventFilter(self, obj, event):
        if obj == self.buttonBox.button(QDialogButtonBox.Ok):
            if event.type() == QEvent.MouseButtonPress:
               mouseEvent = QMouseEvent(event)
               if mouseEvent.buttons() == Qt.LeftButton:
                   src = self.lineEdit_src.text()
                   dst = self.lineEdit_dst.text()
                   if len(src) == 0 or len(dst) == 0:
                       QMessageBox.critical(
                           self, "错误",
                           "文件或路径不允许为空", QMessageBox.Ok
                       )

                       if len(src) == 0:
                           self.lineEdit_src.setFocus()
                       elif len(dst) == 0:
                           self.lineEdit_dst.setFocus()
                       else:
                           pass

                       return True

               if mouseEvent.buttons() == Qt.RightButton:
                   return True

        return QWidget.eventFilter(self, obj, event)

    def on_btnclick_src(self):
        src = QFileDialog.getOpenFileName(
            None,
            "选择文件",
            "./",
            "Microsoft Excel(*.xlsx *.xls)"
        )

        file = src[0]
        self.lineEdit_src.setText(file)
        return

    def on_btnclick_dst(self):
        dir = QFileDialog.getExistingDirectory(
            None,
            "保存到",
            "./"
        )
        self.lineEdit_dst.setText(dir)
        return


    def on_accepted(self):
        src = self.lineEdit_src.text()
        dst = self.lineEdit_dst.text()

        (path, fname) = os.path.split(src)
        (basename, _) = os.path.splitext(fname)
        dst_tile = r"{}/{}_tile.xlsx".format(dst, basename)
        dst_split = r"{}/{}_split.xlsx".format(dst, basename)

        try:
            data = pd.read_excel(src, sheet_name="Sheet1")
            e = Export(_, dst_tile, dst_split)
            e.to_excel_tile(data)
        except Exception as e:
            err = str(e)
            QMessageBox.critical(self, "失败", err, QMessageBox.Ok)

        return

    def on_rejected(self):
        return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win_main = Sersmart()

    # 居中显示
    fg = win_main.frameGeometry()
    point = QDesktopWidget().availableGeometry().center()
    fg.moveCenter(point)
    win_main.move(fg.topLeft())

    win_main.show()
    sys.exit(app.exec_())# author:liao.zhicheng
# date:2021/07/16


import os
import sys
import time
import json
import yaml
import itertools
import re
import pandas as pd
import platform
import socket
import signal
import multiprocessing as mp
import threading
from itertools import groupby

from PyQt5.QtCore import (
         Qt,
         QCoreApplication,
         QObject,
         QSize,
         QRect,
         QThread,
         pyqtSignal,
         QMutex,
         QSemaphore,
         QEvent,
         QRegExp
    )

from PyQt5.QtGui import (
         QMouseEvent,
         QIntValidator,
         QRegExpValidator
    )


from PyQt5.QtWidgets import (
         QApplication,
         QWidget,
         QDesktopWidget,
         QMainWindow,
         QDialog,
         QDialogButtonBox,
         QMessageBox,
         QFileDialog,
         QLineEdit,
         QSizePolicy,
         QCheckBox
    )

from connection import *
from const import *
from logger import *

from ui_mainwindow import Ui_MainWindow
from ui_excelconvert import Ui_Dialog_excelconvert


def time_delay(delay):
    '''
    delay: second
    '''
    timedelta = 0
    time_start = time.time()
    while (timedelta < delay):
        cnt = 0
        while (cnt < 10000):
            cnt += 1
        time_end = time.time()
        timedelta = time_end - time_start
    return

def time_sleep(connection, time):
    '''
    time : second
    '''
    unit = 5
    if time < unit:
       time_delay(time)
       return
    else:
        cnt = 1
        while (cnt * unit < time):
            time_delay(unit)
            connection.write('\n')
            cnt += 1
        else:
            time_delay(time - (cnt - 1) * unit)
    return


mutex = QMutex()
class PrbsProcThread(QThread):
    MAX_PORT_PER_UNIT = 2

    def __init__(self, config=None, login=None, port=0, queue=None):
        super().__init__()

        self.login = login
        self.config = config
        self.slot = login.slot
        self.port = port
        self.queue = queue

        self.finished_records = None
        self.total_cnt = self.calc_total_round_cnt(config)
        self.left_cnt = 0
        self.semaphore = QSemaphore(1)
        self.logger = Logger("sersmart_log_slot{}.txt".format(self.slot)).logger

    @property
    def name(self):
        return self.objectName()

    def run(self):
        self.start_prbs(self.port)

    def update_login(self, login):
        '''
        self.login = login
        '''
        pass

    def update_connection(self):
        '''
        self.con = self.login.con
        '''
        pass

    def exe_commands(self, commands, sleep):
        for command in commands:
            logger.info("{}: {}\n".format(self.name, command))
            self.login.write("{}\n".format(command))
            time_delay(sleep)
            print("{}:\n{}".format(self.name, self.login.read(method=c.READ_VERY_EAGER)))
        return

    def port2unit(self, port=0):
        '''
        编排规则：
        一个unit共2个port， 每个port 4根serdes，serdes编号范围：port0：0 - 3， port1：4-7
        c600共1个unit, port取值: 0 - 1
        c89e共3个unit, port取值：0 - 5
        '''
        unit = port // self.MAX_PORT_PER_UNIT
        return unit

    def calc_total_round_cnt(self, config):
        '''
        计算总共需要循环的次数
        '''
        target = config['target']
        opt_tx = config['opt_tx']
        tx_ce0 = config['ffe']['txce0']
        tx_ce1 = config['ffe']['txce1']
        tx_ce2 = config['ffe']['txce2']

        total_cnt = 0
        ports = target[self.slot]
        for i in itertools.product(ports, opt_tx, tx_ce0, tx_ce1, tx_ce2):
            total_cnt = total_cnt + 1
        return total_cnt

    def init_global_param(self, xfwdp, port):
        print("{}: begin to init global praram.".format(self.name))
        unit = self.port2unit(port)
        cmd = ["gal_debug_disable(7)",
               "diagMsSerdesSetDetectControlOff()",
               "fpp_lif_adeq_stop({}, 0xff)".format(unit)]
        self.exe_commands(cmd, 0.5)

        # 换页操作，将光模块寄存器PAGE0 BYTE127写0x03，换页到PAGE3
        print("{}: begin to init page.".format(self.name))
        cmd = ["data=0x03",
               "BSP_CHIP_OptI2cWrite(1, 0, 127, &data, 1)",
               "diagbspchipOptI2cRead(1, 0, 127)"]
        self.exe_commands(cmd, 0.5)

        cmd = ["fpp_serdes_hrst_set({}, 1, 1)".format(unit),
               "fpp_serdes_hrst_set({}, 1, 0)".format(unit),
               "fpp_serdes_single_init({}, 30, 6)".format(unit)]
        self.exe_commands(cmd, 0.5)

        if xfwdp:
            print("{}: begin to reverse polarity, xfwdp only.".format(self.name))
            self.reverse_polarity(port)

    def reverse_polarity(self, port):
        '''
        极性反转,仅XFWDP需要执行
        '''
        unit = self.port2unit(port)
        cmd = [
            "fpp_serdes_tx_data_inv_en_set({}, 0, 0)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 1, 0)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 2, 0)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 3, 1)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 4, 1)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 5, 0)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 6, 0)".format(unit),
            "fpp_serdes_tx_data_inv_en_set({}, 7, 0)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 0, 1)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 1, 1)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 2, 1)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 3, 0)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 4, 1)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 5, 1)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 6, 1)".format(unit),
            "fpp_serdes_rx_data_inv_en_set({}, 7, 0)".format(unit)
        ]
        self.exe_commands(cmd, 0.5)
        return

    def optical_equalize(self, opt_tx):
        '''
        设置光模块均衡
        '''
        print("{}: begin to set optical module equalization.".format(self.name))
        cmd = [
            "data={}".format(opt_tx),
            "BSP_CHIP_OptI2cWrite(1, 0, 234, &data, 1)",
            "diagbspchipOptI2cRead(1, 0, 234)",
            "BSP_CHIP_OptI2cWrite(1, 0, 235, &data, 1)",
            "diagbspchipOptI2cRead(1, 0, 235)"
        ]
        self.exe_commands(cmd, 0.5)
        return

    def init_serdes_ffe(self, port, ce0, ce1, ce2, serdes):
        '''
        设置FFE参数
        '''
        unit = self.port2unit(port)
        print("{}: begin to set FFE parameters.".format(self.name))
        for serdes_id in serdes:
            self.login.write("fpp_serdes_ffe_set_test({}, {}, {}, {}, {})\n".format(unit, serdes_id, ce0, ce1, ce2))
            time_delay(0.5)
            self.logger.info("{}: {}".format(self.name, self.login.read_buf()))

    def clear_previous_prbs_error_flag(self, port, serdes):
        #先发送一次PRBS
        print("{}: begin to clear previous prbs error flag".format(self.name))
        print("{}: firstly, send prbs signal".format(self.name))

        unit = self.port2unit(port)
        for serdes_id in serdes:
            self.login.write("fpp_serdes_prbs_gen_en_set({}, {}, 7, 1)\n".format(unit, serdes_id))
            time_delay(0.5)
            self.logger.info(self.login.read_buf())

        # 等待2s
        print("{}: waitfor 2s...".format(self.name))
        time_delay(2)

        # 清除误码, 该组命令每1s执行一次, 若干次, 确保无误码
        clr_ef_cnt = self.config['clear_errorflag_count']
        print("{}: secondly, read errorflag, operates {} times.".format(self.name, clr_ef_cnt))
        for i in range(clr_ef_cnt):
            print("{}: {} times".format(self.name, i + 1))
            for serdes_id in serdes:
                self.login.write("fpp_serdes_prbs_chk_en_set({}, {}, 7, 1)\n".format(unit, serdes_id))
                time_delay(0.5)
                self.logger.info(self.login.read_buf())

    def generate_one_record(self, port, opt_tx, tx_ce0, tx_ce1, tx_ce2, serdes_error):
        row = {'port': port, 'opt_tx': opt_tx, "tx_ce0": tx_ce0, "tx_ce1": tx_ce1, "tx_ce2": tx_ce2}
        if '1' in serdes_error:
            row['error'] = 1
        elif 'E' in serdes_error:
            row['error'] = -1
        else:
            row['error'] = 0
        return row

    def start_prbs(self, port):
        xfwdp  = self.config['xfwdp']
        opt_tx = self.config['opt_tx']
        tx_ce0 = self.config['ffe']['txce0']
        tx_ce1 = self.config['ffe']['txce1']
        tx_ce2 = self.config['ffe']['txce2']

        prbs_df = pd.DataFrame(columns=['port', 'opt_tx', 'tx_ce0', 'tx_ce1', 'tx_ce2', 'error'])
        self.finished_records = self.read_finished_records(port)

        print("{}: begin to start prbs, port {}...".format(self.name, port))
        print("{}: there will be {} times to loop for this slot".format(self.name, self.total_cnt))

        serdes_range = list(range(0,8))
        if port % self.MAX_PORT_PER_UNIT == 0:
            serdes = serdes_range[0:4]
        else:
            serdes = serdes_range[4:8]

        mutex.lock()
        print("{}: ++++++++++ mutex locked! ++++++++++".format(self.name))
        self.init_global_param(xfwdp, port)
        mutex.unlock()
        print("{}: ++++++++++ mutex released! ++++++++++".format(self.name))

        cnt = 0
        for opt in opt_tx:
            for ce0, ce1, ce2 in itertools.product(tx_ce0, tx_ce1, tx_ce2):
                if self.finished_records is not None:
                    condition = "port=={} & opt_tx=={} & tx_ce0=={} & tx_ce1=={} & tx_ce2=={}"\
                                .format(port, opt, ce0, ce1, ce2)
                    sel = self.finished_records.query(condition)
                    if self.finished_records.shape[0] > 0 and sel.shape[0] > 0:
                        prbs_df = prbs_df.append(sel, ignore_index=True)
                        print(
                                "{}: slot:{} port:{} opt_tx:{} tx_ce0:{} tx_ce1:{} tx_ce2:{}, "\
                                "already finished, bypass!"\
                                .format(self.name, self.slot, port, opt, ce0, ce1, ce2)
                        )
                        index = sel.index.tolist()
                        self.finished_records.drop(index, inplace=True)

                        cnt = cnt + 1
                        self.left_cnt = self.total_cnt - cnt

                        continue
                    else:
                        row = self.exec_prbs(port, opt, ce0, ce1, ce2, serdes)
                else:
                    row = self.exec_prbs(port, opt, ce0, ce1, ce2, serdes)

                print("{}: to append row to dataframe".format(self.name))
                prbs_df = prbs_df.append(pd.Series(row), ignore_index=True)
                print("{}: done!".format(self.name))
                print("{}: slot{}: current dataframe:\n{}".format(self.name, self.slot, prbs_df))

                # 每次测试完成后都进行保存, 避免测试结果中途丢失
                self.export(port, prbs_df)

                cnt = cnt + 1
                self.left_cnt = self.total_cnt - cnt
                print("{}: slot {} port {}: loop {} complete, there should be {} left!\n"
                      .format(self.name, self.slot, port, cnt, self.left_cnt))

        print("{}:\n----------prbs 100G loop result----------".format(self.name))
        print("{}:\n{}".format(self.name, prbs_df))
        print("{}: port {}: all parameters looped, total count {}, test complete!".format(self.name, port, cnt))

        return prbs_df

    def exec_prbs(self, port, opt, ce0, ce1, ce2, serdes):
        print("{}: start a new loop, port:{},opt:{},ce0:{},ce1:{},ce2{}"\
              .format(self.name, port, opt, ce0, ce1, ce2))
        serdes_error = self.exec_prbs_start(port, opt, ce0, ce1, ce2, serdes)
        print("{}: serdes errorflag: {}".format(self.name, serdes_error))

        print("{}: to generate one record".format(self.name))
        row = self.generate_one_record(port, opt, ce0, ce1, ce2, serdes_error)
        print("{}: row:{}".format(self.name, row))

        print("{}: to stop prbs test".format(self.name))
        self.exec_prbs_stop(port, serdes)
        print("{}: done!".format(self.name))

        return row

    def exec_prbs_start(self, port, opt, ce0, ce1, ce2, serdes):
        mutex.lock()
        print("{}: ++++++++++ mutex locked! ++++++++++".format(self.name))
        self.init_serdes_ffe(port, ce0, ce1, ce2, serdes)
        self.clear_previous_prbs_error_flag(port, serdes)
        mutex.unlock()
        print("{}: ++++++++++ mutex released! ++++++++++".format(self.name))

        # 静默一个时间段, 然后进行PRBS测试, 并检查结果
        wait_time  = self.config['wait_time']
        print("{}: waitfor {}s...".format(self.name, wait_time))
        time_delay(wait_time)

        mutex.lock()
        print("{}: ++++++++++ mutex locked! ++++++++++".format(self.name))
        print(
             "{}: begin to start prbs:"\
             "slot{}, port {}, opt {}, ce0 {}, ce1 {}, ce2 {}"\
             .format(self.name, self.slot, port, opt, ce0, ce1, ce2)
        )

        self.login.clearbuf()

        serdes_error=[]
        recover_cnt = 0
        id_index = 0
        unit = self.port2unit(port)
        while id_index < len(serdes):
            try:
                self.login.write("fpp_serdes_prbs_chk_en_set({}, {}, 7, 1)\n"
                               .format(unit, serdes[id_index]))
                time_delay(0.5)
                buf = self.login.read_buf()
                print("{}: {}".format(self.name, buf))

                try_cnt = 0
                while buf.find('Pattern checker errorflag') < 0:
                   time_delay(0.5)
                   buf = self.login.read_buf()
                   try_cnt += 1
                   if try_cnt > 10:
                       logger.info("{}: Pattern checker errorflag not found, tried {} times"\
                                   .format(self.name, try_cnt))
                       raise Exception("fpp_serdes_prbs_chk_en_set not respond correctly!")
                       break

                res_str = buf.splitlines(False)
                # res_str[0]: 命令本身, res_str[1]:errorflag, rest_str[2]:error_number
                error_code = re.findall(r'Pattern checker errorflag         :0x([0-1])', res_str[1])
                serdes_error.append(error_code[0])
            except Exception as e:
                    self.logger.error("{}: {}".format(self.name, repr(e)))
                    self.login.reconnect()
                    #self.update_connection()

                    # 清空缓存, 不然内容残留,会导致出错上面的buf.splitlines多出一些内容,
                    # 导致解析出错
                    self.login.clearbuf()
                    # 网络未知异常, 放在这里处理, 避免一次出现问题导致后续
                    # 终止的情况, 尝试2次后还不成功, 结果统一填写E
                    if recover_cnt < 2:
                        id_index -= 1
                        recover_cnt += 1
                    else:
                        self.logger.error(str(e))
                        serdes_error.append("E")
            finally:
                   id_index += 1

        mutex.unlock()
        print("{}: ++++++++++ mutex released! ++++++++++".format(self.name))

        return serdes_error

    def exec_prbs_stop(self, port, serdes):
        unit = self.port2unit(port)
        for serdes_id in serdes:
            self.login.write("fpp_serdes_prbs_gen_en_set({}, {}, 7, 0)\n".format(unit, serdes_id))
            time_delay(0.5)
            buf = self.login.read_buf()
            print(buf)
        self.login.clearbuf()
        return

    def read_finished_records(self, port):
        file_orig = "result_orig_{}_{}.xlsx".format(self.slot, port+1)
        file_tile = "result_tile_{}_{}.xlsx".format(self.slot, port+1)
        file_split = "result_split_{}_{}.xlsx".format(self.slot, port+1)
        export = Export(file_orig, file_tile, file_split)
        return export.from_excel()

    def export(self, port, result):
        file_orig = "result_orig_{}_{}.xlsx".format(self.slot, port+1)
        file_tile = "result_tile_{}_{}.xlsx".format(self.slot, port+1)
        file_split = "result_split_{}_{}.xlsx".format(self.slot, port+1)
        export = Export(file_orig, file_tile, file_split)
        export.to_excel(result)


class Login(Telnet):
    def __init__(self, system, ip, port, slot):
        super(__class__, self).__init__(ip, port)

        self.system = system
        self.ip = ip
        self.port = port
        self.slot = slot

    def update_connection(connection):
        #self.con = connection
        pass

    def login_board_c600(self):
        ip = "168.1.{}.0 10000".format(129 + self.slot)
        error_str = "login slot {} error".format(self.slot)

        self.write('\n\n')
        time_delay(1)
        buf = self.read_buf()
        if buf.find("/ #") > 0:
            # 已经在主控shell下
            pass
        elif buf.find('\[FTMETHLP\]#'):
            # 如果在主控shell下，直接telnet到线卡
            # 如果线卡shell下,先exit当前线卡,再telnet到其他线卡
            self.write("exit\n")
        else:
            # 既不在主控shell，也不在线卡shell([FTMETHLP])
            raise Exception(error_str)

        self.write('\n\n')
        if self.waitfor('/ #', 2) == False:
            raise Exception(error_str)
        self.write("telnet {}\n".format(ip))

        if self.waitfor('login:', 2) == False:
            raise Exception(error_str)
        self.write("zxos\n")

        if self.waitfor('password:', 2) == False:
            raise Exception(error_str)
        self.write("zxos{}fnscp@$%\n".format(self.slot))

        if self.waitfor('Successfully login into ushell', 2) == False:
            raise Exception(error_str)
        self.write("\n")

        if self.waitfor('\[admin\]', 2) == False:
            raise Exception(error_str)
        self.write("shell ftm\n")

        if self.waitfor('Now switch to FTMETHLP shell', 2) == False:
            raise Exception(error_str)
        self.write("\n")

        if self.waitfor('\[FTMETHLP\]#', 2) == False:
            raise Exception(error_str)
        self.write("\n")
        return

    def login_board_c89e(self):
        ip = "168.0.{}.1 10000".format(129 + self.slot)
        error_str = "loggin slot {} error".format(self.slot)

        self.write('\n\n')
        time_delay(1)
        buf = self.read_buf()
        if buf.find("/ #") > 0:
            # 已经在主控shell下
            pass
        elif buf.find('\[FTMETHLP\]#'):
            # 如果在主控shell下，直接telnet到线卡
            # 如果线卡shell下,先exit当前线卡,再telnet到其他线卡
            self.write("exit\n")
        else:
            # 既不在主控shell，也不在线卡shell([FTMETHLP])
            raise Exception(error_str)

        self.write('\n\n')
        if self.waitfor('/ #', 2) == False:
            raise Exception(error_str)
        self.write("telnet {}\n".format(ip))

        if self.waitfor('login:', 2) == False:
            raise Exception(error_str)
        self.write("zte\n")

        if self.waitfor('password:', 2) == False:
            raise Exception(error_str)
        self.write("zte\n")

        if self.waitfor('Successfully login into ushell', 2) == False:
            raise Exception(error_str)
        self.write("\n\n\n")

        if self.waitfor('\[admin\]', 2) == False:
            raise Exception(error_str)
        self.write("shell fcm\n")

        if self.waitfor('Now switch to FCMMGRPF_F shell', 2) == False:
            raise Exception(error_str)
        self.write("\n")

        if self.waitfor('\[FCMMGRPF_F\]#', 2) == False:
            raise Exception(error_str)
        self.write("\n")
        return

    def connect_board(self):
        if self.system == 'c600':
            self.login_board_c600()
        elif self.system == 'c89e':
            self.login_board_c89e()
        return

    def reconnect(self):
        self.close()
        logger.info("reconnect slot {}".format(self.slot))

        cnt = 0
        total_try_times = 10000
        while cnt < total_try_times:
            try:
                self.open(self.ip, self.port)
                logger.info("{} times has been tried, succeed".format(cnt))
                break
            except Exception as e:
                logger.error("reconnect {} times, failed!".format(cnt))
                logger.error(str(e))
                logger.error(repr(e))

                cnt += 1
                time_delay(10)
        else:
            raise Exception("NetworkBlock")
            return

        self.connect_board()
        return


class Export():
    def __init__(self, file_orig, file_tile, file_split):
        self.file_orig = file_orig
        self.file_tile = file_tile
        self.file_split = file_split

    def to_excel(self, result):
        if len(result) == 0 :
            logger.error("no data in result dataframe!")
            return

        logger.info("begin to export to xlsx....")
        try:
            writer = pd.ExcelWriter(self.file_orig, engine='xlsxwriter')
            result.to_excel(writer, sheet_name='Sheet1', index=False)

            workbook = writer.book
            worksheet = writer.sheets['Sheet1']

            # 表头上色
            row_header_format = workbook.add_format({
                'bg_color': '#CEE3F6',
                'bold':  True,
                'border': 1
                }
            )
            for col_num, value in enumerate(result.columns.values):
                worksheet.write(0, col_num, value, row_header_format)

            # 错误标识上色
            error_format = workbook.add_format({
                'bg_color': 'red',
                'font_color': 'black'})
            worksheet.conditional_format(1, 5, len(result), 5, {
                'type':     'text',
                'criteria': 'containing',
                'value':    '1',
                'format':    error_format
                }
            )

            writer.save()
            writer.close()
            logger.info("done!")
        except Exception as e:
            logger.info("failed, error:{}".format(str(e)))
        return

    def to_excel_tile(self, result):
        if len(result) == 0 :
            logger.error("no data in result dataframe!")
            return

        logger.info("begin to export split result to xlsx....")
        result.set_index(['port','opt_tx','tx_ce0', 'tx_ce1', 'tx_ce2'], inplace=True)
        result = result.unstack(fill_value=0)
        result[result == 1] = 'F'

        # 多级索引平铺为一张表格
        row_margin, col_margin, row_offset = [5, 5, 1]
        with pd.ExcelWriter(self.file_tile) as writer:
            result_tile = result.style.set_properties(**{'text-align': 'center'})
            result_tile.to_excel(writer)

            workbook = writer.book
            worksheet = writer.sheets['Sheet1']

            row_num, col_num = result.shape
            error_format = workbook.add_format({
                'bg_color': 'red',
                'font_color': 'black'})
            worksheet.conditional_format(*[0, 0, row_num + row_margin, 20], {
                'type':     'text',
                'criteria': 'containing',
                'value':    'F',
                'format':    error_format})

            worksheet.set_column(4, 14, 6)

            writer.save()
            writer.close()

        # 解多级索引, 拆封为多张表格
        # 取得前3级引用的value
        row_index_port = result.index.get_level_values(0).drop_duplicates()
        row_index_opt_tx = result.index.get_level_values(1).drop_duplicates()
        row_index_tx_ce0 = result.index.get_level_values(2).drop_duplicates()

        with pd.ExcelWriter(self.file_split) as writer:
            for i, j, k in itertools.product(row_index_port, row_index_opt_tx, row_index_tx_ce0):
                # 解多级索引, 留下tx_ce1, txce2
                df_txce_1_0 = result.loc[(i, j, k)]

                row_num, col_num = df_txce_1_0.shape
                df_txce_1_0 = df_txce_1_0.style.set_properties(**{
                    'text-align':  'center',
                    'font-family': 'Times New Roman'})
                df_txce_1_0.to_excel(writer, sheet_name='Sheet1', startrow=row_offset, startcol=0)

                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                error_format = workbook.add_format({
                    'bg_color': 'red',
                    'font_color': 'black'})

                first_row, first_col = [row_offset, 0]
                last_row, last_col = [row_offset + row_num + row_margin,  col_num + col_margin]
                worksheet.conditional_format(first_row, first_col, last_row, last_col, {
                    'type':     'text',
                    'criteria': 'containing',
                    'value':    'F',
                    'format':    error_format})

                worksheet.set_column(0, 0, 10)
                worksheet.set_column(1, col_num + 1, 6)

                comment_row, comment_col = [row_offset - 1, 0]
                cell_format = workbook.add_format({
                    'bold':       True,
                    'font_color': 'black',
                    'bg_color':   '#B8CCE4',
                    'font_name':  'Times New Roman'})

                for m in range(col_num + 1):
                    worksheet.write(comment_row, m, None, cell_format)
                worksheet.write(comment_row, comment_col,
                     "port:{}, opt_tx:{}, tx_ce0:{}".format(i, j, k),
                     cell_format)

                row_offset += row_num + row_margin

            writer.save()
            writer.close()
            logger.info("done!")
            return

    def from_excel(self):
        try:
            data = pd.read_excel(self.file_orig, sheet_name="Sheet1")
        except Exception as e:
            logger.error(str(e))
            logger.error(repr(e))
            data = None
        finally:
            return data


def exit(signum, frame):
    logg.info('sersmart terminated by user\n')
    os._exit(0)
    return


class MsgDispathThread(QThread):
    signal = pyqtSignal(str)

    def __init__(self, queue, parent=None):
        QThread.__init__(self, parent)
        self.queue = queue

    def run(self):
        while True:
            if self.queue.empty():
                continue

            text = self.queue.get(True)
            self.signal.emit(text)
        return


class Sersmart(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Sersmart, self).__init__(parent)

        self.config = None
        self.login = []
        self.threads = []

        self.lineEdit_target = {}
        self.setupUi(self)

        self.setFixedSize(self.width(), self.height())

        self.slotValidator = QRegExpValidator(self)
        self.slotValidator.setRegExp(QRegExp('([1-9]|1[0-9]|20)'))
        self.lineEdit_slot_0.setValidator(self.slotValidator)

        self.portValidator = QRegExpValidator(self)
        self.portValidator.setRegExp(
            QRegExp(
               '(([1-5]-[2-6],)|([1-6],)){,6}(([1-5]-[2-6],)|([1-6],)){,6}'
            )
        )
        self.lineEdit_port_0.setValidator(self.portValidator)

        self.queue = mp.Queue()
        self.msg_receive = MsgDispathThread(self.queue)
        self.msg_receive.signal.connect(self.set_browse_text)
        self.msg_receive.start()

        self.pushButton_startprbs.clicked.connect(self.on_btnclick_start_prbs)
        self.pushButton_stopprbs.clicked.connect(self.on_btnclick_stop_prbs)
        self.pushButton_addtarget.clicked.connect(self.add_target)
        self.pushButton_deltarget.clicked.connect(self.del_target)

        self.init_action()

    def init_action(self):
        self.openFileAction.triggered.connect(self.open_file)
        self.exitAction.triggered.connect(self.exit_app)
        self.actionExcelConvertor.triggered.connect(self.convert_excel)


    def init_config():
        cur_path = os.path.abspath(".")
        config_file = os.path.join(cur_path, c.CONFIG_FILE)

        if platform.system() == 'Windows':
            config_file = '\\\\'.join(config_file.split('\\'))

        try:
            with open(config_file, mode='r',encoding="UTF-8") as f:
                config = yaml.load(f.read())
        except Exception as e:
             logger.info("{} cannot be opened, reject to continue!".format(c.CONFIG_FILE))
             os._exit(0)

        finally:
            return config

    def check_config(config):
        for port in config['port']:
            if port not in range(0, 6):
                logger.error("error portid config!")
                os._exit(0)
        return


    def set_browse_text(self, text):
        self.textBrowser_output.append(text)
        # 实时刷新
        #QApplication.processEvents()


    def open_file(self):
        fname = QFileDialog.getOpenFileName(self, '打开新文件',"","*.yaml")
        #fname[0]:实际文件名,包含路径, fname[1]:文件后缀名
        if fname[0] == '':
            return

        try:
            with open(fname[0], mode='r',encoding="UTF-8") as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
                self.comboBox_system.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.critical(self, "异常", str(e), QMessageBox.Ok)
            return

        if self.config is None:
            return

        index = 0 if self.config['system'] == 'c600' else 1
        self.comboBox_system.setCurrentIndex(index)

        self.lineEdit_ip.setText(self.config['ip'])

        state = Qt.Checked if self.config['xfwdp'] == True else Qt.Unchecked
        self.checkBox_xfwdp.setCheckState(state)

        text = str(self.config['opt_tx']).strip('[').strip(']')
        self.lineEdit_opt_tx.setText(text)

        text = self.pack_numbers(self.config['ffe']['txce0'])
        self.lineEdit_txce0.setText(text)

        text = self.pack_numbers(self.config['ffe']['txce1'])
        self.lineEdit_txce1.setText(text)

        text = self.pack_numbers(self.config['ffe']['txce2'])
        self.lineEdit_txce2.setText(text)

        target = self.config['target']
        slots = list(target.keys())
        # 增加slot, port文本框, 第一行文本框已经在UI中静态添加
        for i in range(1, len(slots)):
            self.add_target()

        # 写入配置
        a = self.lineEdit_target
        b = self.gridLayout_target
        for i in range(0, len(slots)):
            slot = slots[i]
            if i==0:
               self.lineEdit_slot_0.setText(str(slot))
               self.lineEdit_port_0.setText(str(target[slot]).strip('[').strip(']'))
            else:
               a['{}-{}'.format(i+1, 0)].setText(str(slot))
               # ports:
               a['{}-{}'.format(i+1, 1)].setText(str(target[slot]).strip('[').strip(']'))

        text = str(self.config['wait_time'])
        self.lineEdit_wait_time.setText(text)
        text = str(self.config['clear_errorflag_count'])
        self.lineEdit_clr_cnt.setText(text)

        return

    def pack_numbers(self, a):
        fun = lambda x: x[1]-x[0]
        for k, g in groupby(enumerate(a), fun):
            l1 = [j for i, j in g]
            if len(l1) > 1:
                scop = str(min(l1)) + '-' + str(max(l1))
            else:
                scop = l1[0]
        return scop

    def expand_packed_range(self, scope):
        num = []
        first_layer = scope.split(',')
        for numstr in first_layer:
            if '-' not in numstr:
                num.append(int(numstr))
                continue

            second_layer = numstr.split('-')
            if len(second_layer) > 2    :
                QMessageBox.critical(self, "error", "检查配置", QMessageBox.Ok)
                return 0xffff

            if len(second_layer) == 2:
                # 比如:'1-15'转换为1, 2, ...
                i = second_layer[0]
                j = second_layer[1]
                for n in range(int(i), int(j) + 1):
                    num.append(n)
            else:
                # 比如:'1'转换为1
                num.append(int(second_layer[0]))

        return num


    def add_target(self):
        a = self.lineEdit_target
        b = self.gridLayout_target
        bottom_margin = 5
        b.setContentsMargins(0, 0, 0, bottom_margin)

        row = b.rowCount()
        if (row > 3):
            QMessageBox.critical(self, "warning", "超过最大支持数", QMessageBox.Ok)
            return

        w_slot = self.lineEdit_slot_0.width()
        h_slot = self.lineEdit_slot_0.height()
        height = h_slot + bottom_margin

        # 现有控件下移
        pos_x = self.pushButton_addtarget.geometry().x()
        pos_y = self.pushButton_addtarget.geometry().y()
        self.pushButton_addtarget.move(pos_x , pos_y + height)

        pos_x = self.pushButton_deltarget.geometry().x()
        pos_y = self.pushButton_deltarget.geometry().y()
        self.pushButton_deltarget.move(pos_x , pos_y + height)

        pos_x = self.checkBox_xfwdp.geometry().x()
        pos_y = self.checkBox_xfwdp.geometry().y()
        self.checkBox_xfwdp.move(pos_x , pos_y + height)

        # 创建和第一行尺寸一样的控件,分别放在0, 1, 2三列
        edit_slot = QLineEdit(self.gridLayoutWidget)
        edit_slot.setFixedSize(w_slot, h_slot)
        edit_slot.setValidator(self.slotValidator)
        a["{}-{}".format(row, 0)] = edit_slot

        w_port = self.lineEdit_port_0.width()
        h_port = self.lineEdit_port_0.height()
        edit_port = QLineEdit(self.gridLayoutWidget)
        edit_port.setFixedSize(w_port, h_port)
        edit_port.setValidator(self.portValidator)
        a["{}-{}".format(row, 1)] = edit_port

        w_port = self.checkBox_select_0.width()
        h_port = self.checkBox_select_0.height()
        checkbox_select = QCheckBox(self.gridLayoutWidget)
        checkbox_select.setFixedSize(w_port, h_port)
        a["{}-{}".format(row, 2)] = checkbox_select

        # 将创建的控件加入到gridLayout
        b.addWidget(a["{}-{}".format(row, 0)], row, 0, 1, 1)
        b.addWidget(a["{}-{}".format(row, 1)], row, 1, 1, 1)
        b.addWidget(a["{}-{}".format(row, 2)], row, 2, 1, 1)

        return

    def del_target(self):
        row_cnt = self.gridLayout_target.rowCount()
        print(row_cnt)
        for i in range(2, row_cnt):
            a = self.lineEdit_target
            if a["{}-{}".format(i, 2)].isChecked():
                print("{} is checked".format(i))
                self.gridLayout_target.removeWidget(a["{}-{}".format(i, 0)])
                self.gridLayout_target.removeWidget(a["{}-{}".format(i, 1)])
                self.gridLayout_target.removeWidget(a["{}-{}".format(i, 2)])
            else:
                pass

        return

    def get_config_from_ui(self):
        config = {}

        config['system'] = self.comboBox_system.currentText()
        config['ip'] = self.lineEdit_ip.text()
        config['port'] = 23

        if Qt.Checked == self.checkBox_xfwdp.checkState():
            config['xfwdp'] = True
        else:
            config['xfwdp'] = False

        # slot-port QGridLayout 第一行
        c = config['target'] = {}
        slot = int(self.lineEdit_slot_0.text())
        port_str = self.lineEdit_port_0.text()
        c[slot] = self.expand_packed_range(port_str)

        # slot-port QGridLayout 第一行之后的其他行
        for row in range(2, self.gridLayout_target.rowCount()):
            slot = int(self.lineEdit_target['{}-{}'.format(row, 0)].text())
            port_str = self.lineEdit_target['{}-{}'.format(row, 1)].text()
            c[slot] = self.expand_packed_range(port_str)

        config['opt_tx'] = self.expand_packed_range(self.lineEdit_opt_tx.text())

        ffe = config['ffe'] = {}
        ffe['txce0'] = self.expand_packed_range(self.lineEdit_txce0.text())
        ffe['txce1'] = self.expand_packed_range(self.lineEdit_txce1.text())
        ffe['txce2'] = self.expand_packed_range(self.lineEdit_txce2.text())
        config['wait_time'] = int(self.lineEdit_wait_time.text())
        config['clear_errorflag_count'] = int(self.lineEdit_clr_cnt.text())
        self.config = config

        return config


    def on_btnclick_start_prbs(self):
        if len(self.lineEdit_ip.text()) == 0:
            QMessageBox.critical(self, "错误", "ip不能为空!", QMessageBox.Ok)
            self.lineEdit_ip.setFocus()
            return

        if len(self.lineEdit_slot_0.text()) == 0:
            QMessageBox.critical(self, "错误", "槽位号不能为空!", QMessageBox.Ok)
            self.lineEdit_slot_0.setFocus()
            return


        if len(self.lineEdit_port_0.text()) == 0:
            QMessageBox.critical(self, "错误", "端口号不能为空!", QMessageBox.Ok)
            self.lineEdit_port_0.setFocus()
            return

        config = self.get_config_from_ui()
        if not bool(config):
            QMessageBox.critical(self, "错误", "配置未加载!", QMessageBox.Ok)
            return

        if len(self.lineEdit_opt_tx.text()) == 0:
            QMessageBox.critical(self, "错误", "opt_tx参数不能为空!", QMessageBox.Ok)
            self.lineEdit_opt_tx.setFocus()
            return

        if len(self.lineEdit_txce0.text()) == 0:
            QMessageBox.critical(self, "错误", "tx_ce0参数不能为空!", QMessageBox.Ok)
            self.lineEdit_txce0.setFocus()
            return

        if len(self.lineEdit_txce1.text()) == 0:
            QMessageBox.critical(self, "错误", "tx_ce1参数不能为空!", QMessageBox.Ok)
            self.lineEdit_txce1.setFocus()
            return

        if len(self.lineEdit_txce2.text()) == 0:
            QMessageBox.critical(self, "错误", "tx_ce2参数不能为空!", QMessageBox.Ok)
            self.lineEdit_txce2.setFocus()
            return


        # 按钮变灰
        self.pushButton_startprbs.setEnabled(False)
        self.pushButton_startprbs.setEnabled(True)

        # 每个槽位一个Login, 每个端口一个线程
        target = config['target']
        PRO_NUM = len(target)
        for slot in target.keys():
            # 初始化连接, 每个进程需要一个连接
            login = Login(
                config['system'],
                config['ip'],
                config['port'],
                slot
            )
            self.login.append(login)

            msg = "start to login slot {}".format(slot)
            logger.info(msg)
            self.queue.put(msg)

            try:
                login.connect_board()
                logger.info("successfully!")
            except Exception as e:
                logger.error("failed!")
                logger.error(str(e))
                continue

            for port in target[slot]:
                thread = PrbsProcThread(config, login, port, self.queue, )
                thread.setObjectName("task_{}_{}".format(slot, port))
                self.threads.append(thread)

            for thread in self.threads:
                thread.start()
                time_delay(30)

    def on_btnclick_stop_prbs(self):
        # 开始按钮变亮, 停止按钮变灰
        self.pushButton_startprbs.setEnabled(True)
        self.pushButton_stopprbs.setEnabled(False)

        return

    def convert_excel(self):
        dialog = Excelconvert()
        dialog.lineEdit_src.setFocus()
        dialog.exec_()

    def exit_app(self):
        QCoreApplication.instance().quit()
        return


class Excelconvert(QDialog, Ui_Dialog_excelconvert):
    def __init__(self, parent=None):
        super(Excelconvert, self).__init__(parent)
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())

        self.pushButton_src.clicked.connect(self.on_btnclick_src)
        self.pushButton_dst.clicked.connect(self.on_btnclick_dst)

        self.buttonBox.button(QDialogButtonBox.Ok).installEventFilter(self)
        self.buttonBox.accepted.connect(self.on_accepted)
        self.buttonBox.rejected.connect(self.on_rejected)

    def eventFilter(self, obj, event):
        if obj == self.buttonBox.button(QDialogButtonBox.Ok):
            if event.type() == QEvent.MouseButtonPress:
               mouseEvent = QMouseEvent(event)
               if mouseEvent.buttons() == Qt.LeftButton:
                   src = self.lineEdit_src.text()
                   dst = self.lineEdit_dst.text()
                   if len(src) == 0 or len(dst) == 0:
                       QMessageBox.critical(
                           self, "错误",
                           "文件或路径不允许为空", QMessageBox.Ok
                       )

                       if len(src) == 0:
                           self.lineEdit_src.setFocus()
                       elif len(dst) == 0:
                           self.lineEdit_dst.setFocus()
                       else:
                           pass

                       return True

               if mouseEvent.buttons() == Qt.RightButton:
                   return True

        return QWidget.eventFilter(self, obj, event)

    def on_btnclick_src(self):
        src = QFileDialog.getOpenFileName(
            None,
            "选择文件",
            "./",
            "Microsoft Excel(*.xlsx *.xls)"
        )

        file = src[0]
        self.lineEdit_src.setText(file)
        return

    def on_btnclick_dst(self):
        dir = QFileDialog.getExistingDirectory(
            None,
            "保存到",
            "./"
        )
        self.lineEdit_dst.setText(dir)
        return


    def on_accepted(self):
        src = self.lineEdit_src.text()
        dst = self.lineEdit_dst.text()

        (path, fname) = os.path.split(src)
        (basename, _) = os.path.splitext(fname)
        dst_tile = r"{}/{}_tile.xlsx".format(dst, basename)
        dst_split = r"{}/{}_split.xlsx".format(dst, basename)

        try:
            data = pd.read_excel(src, sheet_name="Sheet1")
            e = Export(_, dst_tile, dst_split)
            e.to_excel_tile(data)
        except Exception as e:
            err = str(e)
            QMessageBox.critical(self, "失败", err, QMessageBox.Ok)

        return

    def on_rejected(self):
        return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win_main = Sersmart()

    # 居中显示
    fg = win_main.frameGeometry()
    point = QDesktopWidget().availableGeometry().center()
    fg.moveCenter(point)
    win_main.move(fg.topLeft())

    win_main.show()
    sys.exit(app.exec_())