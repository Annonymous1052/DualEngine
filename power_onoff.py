import Monsoon.HVPM as HVPM
import Monsoon.sampleEngine as sampleEngine
import Monsoon.Operations as op

Mon = HVPM.Monsoon()
Mon.setup_usb()

Mon.setVout(4.2)  # on

x = input()

if x == 0:
    Mon.setVout(0)  # off
    Mon.closeDevice()
