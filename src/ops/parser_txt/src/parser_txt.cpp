/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-03-15 14:41:38
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-03-18 16:29:27
 * @FilePath: /D2D-placer/src/ops/read_txt/src/read_txt.cpp
 * @Description: 
 */

#include <pybind11/pybind11.h>
// 3d-placer parser
#include "utility/paramHdl.h"
#include "parser/parser.h"
#include "dataModel/dm.h"
#include "placer/placer.h"
#include "include/common.h"

PLACER_BEGIN_NAMESPACE

int parser_txt_forward(pybind11::list const& args)
{
  clock_t tStart = clock();
  // args -> argc, argv
  int argc = pybind11::len(args); 
  char** argv = new char* [argc]; 
  for (int i = 0; i < argc; ++i)
  {
      std::string token = pybind11::str(args[i]); 
      argv[i] = new char [token.size()+1];
      std::copy(token.begin(), token.end(), argv[i]); 
      argv[i][token.size()] = '\0';
  }

  // txt2bookself by 3d-placer
  ParamHdl_C paramHdl = ParamHdl_C(argc, argv);
  Parser_C parser;
  parser.read_file(paramHdl.get_input_fileName());
  DmMgr_C* dmMgr = new DmMgr_C(parser, paramHdl, tStart);
  dmMgr->print_info();
  dmMgr->txt2bookself();
  return 0;
}

PLACER_END_NAMESPACE


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("parser_txt", &PLACER_NAMESPACE::parser_txt_forward, "parser_txt_forward");
}



