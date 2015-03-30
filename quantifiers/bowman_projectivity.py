Ofrom quantgen import *

bowman = {}

bowman['some'] =  [{EQ:EQ, FOR:FOR, REV:REV, NEG:COV, ALT:INDY, COV:COV,  INDY:INDY}, {EQ:EQ, FOR:FOR, REV:REV, NEG:COV, ALT:INDY, COV:COV,  INDY:INDY}]

bowman['all'] = [{EQ:EQ, FOR:REV, REV:FOR, NEG:ALT, ALT:INDY, COV:ALT,  INDY:INDY}, {EQ:EQ, FOR:FOR, REV:REV, NEG:ALT, ALT:ALT,  COV:INDY, INDY:INDY}]

bowman['two'] =  [{EQ:EQ, FOR:FOR, REV:REV, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY}, {EQ:EQ, FOR:FOR, REV:REV, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY}]

bowman['three'] =  [{EQ:EQ, FOR:FOR, REV:REV, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY}, {EQ:EQ, FOR:FOR, REV:REV, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY}]

bowman['no'] =  [{EQ:EQ, FOR:REV, REV:FOR, NEG:ALT, ALT:INDY, COV:ALT,  INDY:INDY}, {EQ:EQ, FOR:REV, REV:FOR, NEG:ALT, ALT:INDY, COV:ALT,  INDY:INDY}]

bowman['not_all'] =  [{EQ:EQ, FOR:FOR, REV:REV, NEG:COV, ALT:INDY, COV:COV,  INDY:INDY}, {EQ:EQ, FOR:REV, REV:FOR, NEG:COV, ALT:COV, COV:INDY,  INDY:INDY}]

bowman['most'] =  [{EQ:EQ, FOR:INDY, REV:INDY, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY}, {EQ:EQ, FOR:FOR, REV:REV, NEG:ALT, ALT:ALT, COV:INDY,  INDY:INDY}]

bowman['not_most'] =  [{EQ:EQ, FOR:INDY, REV:INDY, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY}, {EQ:EQ, FOR:REV, REV:FOR, NEG:ALT, ALT:COV, COV:ALT,  INDY:INDY}]

bowman['lt_two'] =  [{EQ:EQ, FOR:REV, REV:FOR, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY}, {EQ:EQ, FOR:REV, REV:FOR, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY}]

bowman['lt_three'] =  [{EQ:EQ, FOR:REV, REV:FOR, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY}, {EQ:EQ, FOR:REV, REV:FOR, NEG:INDY, ALT:INDY, COV:INDY,  INDY:INDY}]


for key, val in bowman.items():
      print "======================================================================"
      print key, val == projectivity[key]
      if val != projectivity[key]:
            for i, d in enumerate(val):
                  print i
                  for r1, r2 in d.items():
                        if r2 != projectivity[key][i][r1]:
                              print "\t", r1, r2, projectivity[key][i][r1]
                  
            
      
