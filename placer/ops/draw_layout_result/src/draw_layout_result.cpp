/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-09-23 22:54:22
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-09-25 16:04:45
 * @FilePath:
 * /D2D-placer/placer/ops/draw_layout_result/src/draw_layout_result.cpp
 * @Description:
 */

#include <pybind11/pybind11.h>
// 3d-placer parser
#include "dataModel/dm.h"
#include "include/common.h"
#include "parser/parser.h"
#include "placer/placer.h"
#include "utility/paramHdl.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

PLACER_BEGIN_NAMESPACE

void draw_layout_result_forward(pybind11::list const &args) {
  clock_t tStart = clock();
  // args -> argc, argv
  int argc = pybind11::len(args);
  char **argv = new char *[argc];
  for (int i = 0; i < argc; ++i) {
    string token = pybind11::str(args[i]);
    argv[i] = new char[token.size() + 1];
    copy(token.begin(), token.end(), argv[i]);
    argv[i][token.size()] = '\0';
  }

  // txt2bookself by 3d-placer
  ParamHdl_C paramHdl = ParamHdl_C(argc, argv);
  Parser_C parser;
  parser.read_file(paramHdl.get_input_fileName());
  DmMgr_C *dmMgr = new DmMgr_C(parser, paramHdl, tStart);
  dmMgr->print_info();
  dmMgr->draw_layout_result(argv[2]);
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("draw_layout_result", &PLACER_NAMESPACE::draw_layout_result_forward,
        "draw_layout_result_forward");
}