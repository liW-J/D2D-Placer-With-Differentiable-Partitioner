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
#include "utility/aux.h"
#include "utility/paramHdl.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

PLACER_BEGIN_NAMESPACE

namespace {

struct PlacementData {
  Chip_C *chip;
  Design_C *design;
  std::vector<CellLib_C *> cell_libs;
};

std::string join_path(std::string dir, const std::string &name) {
  if (dir.empty()) {
    return name;
  }
  if (dir.back() == '/') {
    return dir + name;
  }
  return dir + "/" + name;
}

std::string shell_quote(const std::string &path) {
  std::string quoted = "'";
  for (char ch : path) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

bool file_exists(const std::string &file_name) {
  std::ifstream input(file_name);
  return input.good();
}

std::string placement_file_or_throw(const std::string &result_dir,
                                    const std::string &subdir,
                                    const std::string &prefix) {
  std::string final_file =
      join_path(result_dir, subdir + "/" + prefix + ".final.pl");
  if (file_exists(final_file)) {
    return final_file;
  }

  std::string ntup_file =
      join_path(result_dir, subdir + "/" + prefix + ".ntup.pl");
  if (file_exists(ntup_file)) {
    return ntup_file;
  }

  std::string gp_file = join_path(result_dir, subdir + "/" + prefix + ".gp.pl");
  if (file_exists(gp_file)) {
    return gp_file;
  }

  throw std::runtime_error("Cannot find placement result for " + prefix +
                           ": tried " + final_file + ", " + ntup_file +
                           " and " + gp_file);
}

PlacementData build_placement_data(Parser_C &parser) {
  std::unordered_map<std::string, int> tech_name_to_id;
  std::unordered_map<std::string, CellLib_C *> cell_lib_by_name;
  std::vector<CellLib_C *> cell_libs;

  std::vector<ParserTech> &techs = parser.get_techs();
  for (int tech_id = 0; tech_id < static_cast<int>(techs.size());
       ++tech_id) {
    ParserTech &tech = techs[tech_id];
    tech_name_to_id[tech.name] = tech_id;

    for (ParserLibCell &lib_cell : tech.v_libCell) {
      CellLib_C *cell_lib = nullptr;
      auto existing = cell_lib_by_name.find(lib_cell.name);
      if (existing == cell_lib_by_name.end()) {
        cell_lib =
            new CellLib_C(lib_cell.name, lib_cell.numLibPin, techs.size(),
                          lib_cell.isMacro);
        cell_lib_by_name.emplace(lib_cell.name, cell_lib);
        cell_libs.emplace_back(cell_lib);
      } else {
        cell_lib = existing->second;
      }

      cell_lib->set_size(tech_id, lib_cell.sizeX, lib_cell.sizeY);
      for (ParserLibPin &lib_pin : lib_cell.v_libPin) {
        cell_lib->add_pin(tech_id, lib_pin.name,
                          Pos(lib_pin.locationX, lib_pin.locationY));
      }
    }
  }

  Design_C *design = new Design_C();
  for (ParserInst &inst : parser.get_insts()) {
    auto lib = cell_lib_by_name.find(inst.libCellName);
    if (lib == cell_lib_by_name.end()) {
      throw std::runtime_error("Unknown lib cell in input: " +
                               inst.libCellName);
    }
    design->add_cell(new Cell_C(inst.name, lib->second));
  }

  auto &cells = design->get_cells_map();
  for (ParserNet &net : parser.get_nets()) {
    Net_C *new_net = new Net_C(net.name);
    for (ParserPin &pin : net.v_pin) {
      auto cell = cells.find(pin.instName);
      if (cell == cells.end()) {
        throw std::runtime_error("Unknown instance in netlist: " +
                                 pin.instName);
      }
      new_net->add_pin(cell->second->get_pin(pin.libPinName));
    }
    design->add_net(new_net);
  }
  design->set_cell_degree();

  ParserDie top_die = parser.get_top_die_info();
  ParserDie bot_die = parser.get_bot_die_info();
  Chip_C *chip =
      new Chip_C(top_die.ur_x - top_die.ll_x, top_die.ur_y - top_die.ll_y, 2);
  chip->set_die(0, top_die.maxUtil, tech_name_to_id[top_die.dieTech],
                top_die.rowHeight);
  chip->set_die(1, bot_die.maxUtil, tech_name_to_id[bot_die.dieTech],
                bot_die.rowHeight);
  Terminal terminal = parser.get_terminal_info();
  chip->set_ball(terminal.sizeX, terminal.sizeY, terminal.spacing);

  for (Cell_C *cell : design->get_cells()) {
    cell->set_die(chip->get_die(0));
  }

  return {chip, design, cell_libs};
}

void load_tier_placement(const std::string &file_name, int die_id, Chip_C *chip,
                         Design_C *design) {
  AUX aux;
  std::vector<AuxNode> placed_nodes;
  if (!aux.read_pl(file_name, placed_nodes)) {
    throw std::runtime_error("Cannot read placement result: " + file_name);
  }

  auto &cells = design->get_cells_map();
  for (AuxNode &node : placed_nodes) {
    auto cell = cells.find(node.name);
    if (cell == cells.end()) {
      continue;
    }
    cell->second->set_die(chip->get_die(die_id));
    cell->second->set_xy(Pos(node.x, node.y));
  }
}

void copy_file_if_exists(const std::string &src, const std::string &dst) {
  std::ifstream in(src, std::ios::binary);
  if (!in.good()) {
    return;
  }
  std::ofstream out(dst, std::ios::binary);
  out << in.rdbuf();
}

} // namespace

void draw_layout_result_forward(pybind11::list const &args) {
  clock_t tStart = clock();
  // args -> argc, argv
  int argc = pybind11::len(args);
  if (argc < 3) {
    throw std::invalid_argument(
        "draw_layout_result expects argv: 3d-placer <input_txt> <result_dir>");
  }

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
  PlacementData data = build_placement_data(parser);

  std::string result_dir = argv[2];
  load_tier_placement(placement_file_or_throw(result_dir, "tier0", "tier0"),
                      0, data.chip, data.design);
  load_tier_placement(placement_file_or_throw(result_dir, "tier1", "tier1"),
                      1, data.chip, data.design);

  Placer_C placer(data.chip, data.design, paramHdl, tStart);
  if (!placer.read_pl_and_set_pos_for_ball(placement_file_or_throw(
          result_dir, "terminal", "terminal"))) {
    throw std::runtime_error("Cannot read terminal placement result under: " +
                             result_dir);
  }
  int hpwl = placer.cal_HPWL();
  std::cout << BLUE << "[draw_layout_result]" << RESET << " - HPWL = "
            << hpwl << "\n";
  placer.init_draw_dir();
  placer.draw_layout_result("-final");

  std::string default_draw_file =
      join_path(join_path("./draw", paramHdl.get_case_name()),
                paramHdl.get_case_name() + "-final.html");
  std::string mkdir_cmd = "mkdir -p " + shell_quote(result_dir);
  std::system(mkdir_cmd.c_str());
  copy_file_if_exists(default_draw_file,
                      join_path(result_dir,
                                paramHdl.get_case_name() + "-final.html"));
}

PLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("draw_layout_result", &PLACER_NAMESPACE::draw_layout_result_forward,
        "draw_layout_result_forward");
}
