/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-03-15 14:41:38
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-06-14 02:57:55
 * @FilePath: /D2D-placer/placer/ops/read_txt/src/read_txt.cpp
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

struct DieSpec {
  int numTechnologies;
  int dieSizeX;
  int dieSizeY;
  int topDieMaxUtil;
  int bottomDieMaxUtil;
  int terminalSizeX;
  int terminalSizeY;
  int terminalSpacing;
};

DieSpec parser_txt_forward(pybind11::list const &args) {
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
  dmMgr->txt2bookself();

  // parameter extraction
  DieSpec params;
  string file_name = paramHdl.get_input_fileName();
  std::ifstream file(file_name);

  if (!file.is_open()) {
    LOG(ERROR, "cannot open file: %s", file_name.c_str());
  }

  std::string line;
  int x1, y1, x2, y2;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string key;
    iss >> key;

    if (key == "NumTechnologies") {
      if (!(iss >> params.numTechnologies)) {
        LOG(ERROR, "NumTechnologies can not be parsed");
      }
    } else if (key == "TopDieMaxUtil") {
      if (!(iss >> params.topDieMaxUtil)) {
        LOG(ERROR, "TopDieMaxUtil can not be parsed");
      }
    } else if (key == "DieSize") {
      if (!(iss >> x1 >> y1 >> x2 >> y2)) {
        LOG(ERROR, "DieSize can not be parsed");
      }
    } else if (key == "BottomDieMaxUtil") {
      if (!(iss >> params.bottomDieMaxUtil)) {
        LOG(ERROR, "BottomDieMaxUtil can not be parsed");
      }
    } else if (key == "TerminalSize") {
      if (!(iss >> params.terminalSizeX >> params.terminalSizeY)) {
        LOG(ERROR, "TerminalSize can not be parsed");
      }
    } else if (key == "TerminalSpacing") {
      if (!(iss >> params.terminalSpacing)) {
        LOG(ERROR, "TerminalSpacing can not be parsed");
      }
    }
  }
  file.close();
  params.dieSizeX = x2 - x1;
  params.dieSizeY = y2 - y1;

  return params;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<placer::DieSpec>(m, "DieSpec")
      .def(pybind11::init<>())
      .def_readwrite("numTechnologies", &placer::DieSpec::numTechnologies)
      .def_readwrite("dieSizeX", &placer::DieSpec::dieSizeX)
      .def_readwrite("dieSizeY", &placer::DieSpec::dieSizeY)
      .def_readwrite("topDieMaxUtil", &placer::DieSpec::topDieMaxUtil)
      .def_readwrite("bottomDieMaxUtil", &placer::DieSpec::bottomDieMaxUtil)
      .def_readwrite("terminalSizeX", &placer::DieSpec::terminalSizeX)
      .def_readwrite("terminalSizeY", &placer::DieSpec::terminalSizeY)
      .def_readwrite("terminalSpacing", &placer::DieSpec::terminalSpacing);

  m.def("parser_txt", &PLACER_NAMESPACE::parser_txt_forward,
        "parser_txt_forward");
}
