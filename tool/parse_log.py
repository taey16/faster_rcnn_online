#!/usr/bin/env python

import os
import sys

FIELD_NAME = {'accuracy_cls': 'accuracy_cls', 
              'loss_cls': 'loss_cls', 
              'rpn_cls_accuracy': 'rpn_cls_accuracy', 
              'rpn_cls_loss': 'rpn_cls_loss', 
              'rpn_loss_bbox': 'rpn_loss_bbox',
              'loss': 'loss'}
def parse_log(filename):
  assert(os.path.exists(filename))

  # NOTE: output log file pointer
  logger_trn = open('%s.trn' % filename, 'w')
  logger_val = open('%s.val' % filename, 'w')

  # NOTE: start to parsing
  with open(filename, 'r') as input_fp:
    phase = ''
    for line in input_fp:
      if line == None: break
      tokens = line.strip().split(' ')
      for i, token in enumerate(tokens):
        # NOTE: get iteration number and phase
        if token == 'Iteration':
          if tokens[i+2] == 'Testing':
            phase = 'val'
            print('Testing %d' % int(tokens[i+1][:-1]))
            logger_val.write('\n%f ' % int(tokens[i+1][:-1]))
            #if token == '=':
            #  print(tokens[i+1])
          elif tokens[i+2] == 'loss':
            phase = 'trn'
            print('Training %d' % int(tokens[i+1][:-1]))
            logger_trn.write('\n%f ' % int(tokens[i+1][:-1]))
            #if token == '=':
            #  print(tokens[i+1])

        # NOTE: save FIELD_NAME values corresponding iteration and phase
        if token == '=':
          if phase == 'trn':
            if FIELD_NAME.has_key(tokens[i-1]):
              print('%s %f' % (FIELD_NAME[tokens[i-1]], float(tokens[i+1])))
              logger_trn.write('%f ' % float(tokens[i+1]))
          elif phase == 'val':
            if FIELD_NAME.has_key(tokens[i-1]):
              print('%s %f' % (FIELD_NAME[tokens[i-1]], float(tokens[i+1])))
              logger_val.write('%f ' % float(tokens[i+1]))

  logger_trn.close()
  logger_val.close()
  
  
if __name__ == '__main__':
  if len(sys.argv) < 2:
    exit()
  else:
    #import pdb; pdb.set_trace()
    # NOTE: input: log-filename
    # output: train, val log file (log_filename.trn, log_filename.val)
    parse_log(sys.argv[1])
