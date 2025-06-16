import Monsoon.HVPM as HVPM
import Monsoon.sampleEngine as sampleEngine
import Monsoon.Operations as op
import time
import subprocess
import csv
import argparse


parser = argparse.ArgumentParser(description='Mobile Power Logger for Monsoon Power Monitor')
parser.add_argument('--name', type=str, default='S21U', help='Output CSV file name (without .csv extension)')
parser.add_argument('--time', type=int, default=100, help='Measurement duration in seconds')
args = parser.parse_args()


csv_filename = f"{args.name}.csv"
measurement_duration = args.time

print(f"Starting power measurement...")
print(f"Output file: {csv_filename}")
print(f"Duration: {measurement_duration} seconds")

Mon = HVPM.Monsoon()
Mon.setup_usb()

engine = sampleEngine.SampleEngine(Mon)
engine.disableCSVOutput()
engine.ConsoleOutput(False)

engine.enableChannel(sampleEngine.channels.MainCurrent)
engine.enableChannel(sampleEngine.channels.MainVoltage)

start = time.time()
prev = start - 1
f = open(csv_filename, 'w', newline='')  
wt = csv.writer(f)
wt.writerow(["Timestamp", "Current(A)", "Voltage(V)", "Power(W)"])

while time.time() - start < measurement_duration:  
    if time.time() - prev >= 1:
        print(time.time()-start)
        prev = time.time()
        engine.startSampling(1)
        sample = engine.getSamples()
        current = sample[sampleEngine.channels.MainCurrent][0]
        voltage = sample[sampleEngine.channels.MainVoltage][0]
        Mon.stopSampling()
        wt.writerow([round(time.time()-start,3), int(current) / 1000, voltage, round(current * voltage / 1000, 5)])
f.close()

engine.disableChannel(sampleEngine.channels.MainCurrent)
engine.disableChannel(sampleEngine.channels.MainVoltage)

print("measurement done!")

import winsound as sd
def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)
beepsound()