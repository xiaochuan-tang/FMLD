# -*- coding: utf-8 -*-
'''
udp可靠传输,支持发送文件和接收文件,发送字符串和接收字符串
'''
import socket
import struct
import zlib

CHUNK_SIZE = 1024  # 假设的数据块大小
TIMEOUT = 5  # 超时时间，单位秒


def calculate_checksum(data):
    """计算数据的校验和"""
    return zlib.crc32(data)


def create_packet(sequence_number, chunk):
    """创建数据包，包含序列号、校验和和数据"""
    checksum = calculate_checksum(chunk)
    header = struct.pack('>I', sequence_number) + struct.pack('>I', checksum)
    return header + chunk


def parse_packet(packet):
    """解析数据包，返回序列号、数据和校验和"""
    sequence_number, checksum = struct.unpack('>II', packet[:8])
    data = packet[8:]
    return sequence_number, data, checksum


def create_ack(sequence_number):
    """创建ACK包"""
    return struct.pack('>I', sequence_number)


def parse_ack(ack):
    """解析ACK包，返回序列号"""
    try:
        sequence_number, = struct.unpack('>I', ack)
    except struct.error:
        return None
    return sequence_number


# 发送方
def send_file(sock, filename, dest_addr):
    f = open(filename, "rb")
    sequence_number = 0
    ack_expected = 0
    window_size = 1
    window = []

    while True:
        if len(window) < window_size:
            chunk = f.read(CHUNK_SIZE)
            if not chunk and len(window) == 0:
                break  # 文件和窗口都为空时结束

            if chunk:
                packet = create_packet(sequence_number, chunk)
                sock.sendto(packet, dest_addr)
                window.append((sequence_number, packet))
                sequence_number += 1
        while window:
            try:
                sock.settimeout(TIMEOUT)
                ack, _ = sock.recvfrom(1024)
                ack = parse_ack(ack)
                # 移除已确认的包
                if ack == None:
                    continue
                window = [(seq, pkt) for seq, pkt in window if seq > ack]
                if len(window) > 0:
                    ack_expected = window[0][0]
                else:
                    ack_expected = sequence_number
            except socket.timeout:
                # 超时，重发窗口内所有包
                print("time out")
                for _, pkt in window:
                    sock.sendto(pkt, dest_addr)

    # 发送结束信号
    send_string(sock, "END", dest_addr)
    window.clear()
    f.close()


def send_string(sock, string_data, dest_addr):
    sequence_number = 0
    ack_expected = 0
    window_size = 1
    window = []

    # 将字符串切分为块
    chunks = [string_data[i:i + 128] for i in range(0, len(string_data), 128)]
    for chunk in chunks:
        if not chunk:
            break  # 如果块为空，则结束循环
        packet = create_packet(sequence_number, chunk.encode())  # 编码字符串为字节串
        sock.sendto(packet, dest_addr)
        window.append((sequence_number, packet))
        sequence_number += 1
        while window:
            try:
                sock.settimeout(TIMEOUT)
                ack, _ = sock.recvfrom(128)
                ack = parse_ack(ack)
                # 移除已确认的包
                if ack == None:
                    continue
                window = [(seq, pkt) for seq, pkt in window if seq > ack]
                if len(window) > 0:
                    ack_expected = window[0][0]
                else:
                    ack_expected = sequence_number
            except socket.timeout:
                # 超时，重发窗口内所有包
                print("time out")
                for _, pkt in window:
                    sock.sendto(pkt, dest_addr)

    # 发送结束信号
    sock.sendto(b"END", dest_addr)
    window.clear()


# 接收方
def recv_file(sock, filename):
    sock.settimeout(None)
    f = open(filename, "wb")
    expected_seq = 0

    while True:
        packet, addr = sock.recvfrom(4096)

        if packet == b"END":
            break
        try:
            seq, data, checksum = parse_packet(packet)
            # print("seq:", seq, "expected_seq", seq)
            if checksum == calculate_checksum(data) and seq == expected_seq:
                f.write(data)
                expected_seq += 1

            # 发送ACK
            sock.sendto(create_ack(expected_seq - 1), addr)
        except (struct.error, socket.timeout) as e:
            continue
        except BlockingIOError:
            time.sleep(0.1)
    f.close()


def recv_string(sock):
    sock.settimeout(None)
    received_data = []
    expected_seq = 0

    while True:
        packet, addr = sock.recvfrom(128)

        # 检查是否是结束信号
        if packet == b"END":
            break

        try:
            seq, data, checksum = parse_packet(packet)
            # print("接收到序列号:", seq, "期望的序列号:", expected_seq)

            if checksum == calculate_checksum(data) and seq == expected_seq:
                # 将数据追加到接收缓冲区
                received_data.append(data)
                expected_seq += 1

            # 发送ACK确认
            sock.sendto(create_ack(seq), addr)
        except (struct.error, socket.timeout) as e:
            print("error")
            continue
        except BlockingIOError:
            time.sleep(0.1)

    # 将接收到的字节序列组合成字符串
    return b''.join(received_data).decode(), addr
