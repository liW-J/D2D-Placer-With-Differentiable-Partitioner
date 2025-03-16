/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-03-15 14:41:38
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-03-16 20:21:48
 * @FilePath: /D2D-placer/src/ops/read_txt/src/read_txt.cpp
 * @Description: 
 */

// dreamplace database dependency
#include "dreamplace/ops/place_io/src/PyPlaceDB.h"
// 3d-placer parser
#include "parser/parser.h"

#include "include/common.h"

PLACER_BEGIN_NAMESPACE

DREAMPLACE_NAMESPACE::PlaceDB parser_txt_forward(string fileName)
{
  DREAMPLACE_NAMESPACE::PlaceDB db; 

  // txt2bookself
  Parser_C parser;
  if(!parser.read_file(fileName))
    return db;
  return db;
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("parser_txt", &PLACER_NAMESPACE::parser_txt_forward, "parser_txt_forward");
}



