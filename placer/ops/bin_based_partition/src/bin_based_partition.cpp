/*
 * @Author: JeanneWillis hi@jeannewillis.cn
 * @Date: 2025-05-18 18:32:13
 * @LastEditors: JeanneWillis hi@jeannewillis.cn
 * @LastEditTime: 2025-05-18 22:00:02
 * @FilePath: /D2D-placer/placer/ops/bin_based_partition/src/bin_based_partition.cpp
 * @Description: 
 */

void Placer_C::bin_based_partition_new() {
    double cutline = 0.5;
    double width_avg0 = 0;
    double width_avg1 = 0;
    for(Cell_C* cell : _vCell){ // Todo v_cell ?= _vCell
        width_avg0 += (double)cell->get_width(_pChip->get_die(0)->get_techId())/ _vCell.size();
        width_avg1 += (double)cell->get_width(_pChip->get_die(1)->get_techId())/ _vCell.size();
    }
    cutline = ((width_avg1*_pChip->get_die(1)->get_row_height()) / (width_avg1*_pChip->get_die(1)->get_row_height() + width_avg0*_pChip->get_die(0)->get_row_height())) * (_pChip->get_die(0)->get_max_util() / _pChip->get_die(1)->get_max_util());
    cout << "cutline = " << cutline << "\n";

    // int bins_num = _vCell.size() * 1.1;
    
    int bins_per_row = 1;
    int bins_per_col = 1;
    _multiLevel = false;
    
    cout << _paramHdl.get_case_name() << ": " << bins_per_row <<"\n";
    int bin_width = _pChip->get_die(0)->get_width() / bins_per_row;
    int bin_height = _pChip->get_die(0)->get_height() / bins_per_col;
    int bin_num = bins_per_row * bins_per_col;
    vector <vector <vector <Cell_C*>>> bins(bins_per_row, vector< vector <Cell_C*>> (bins_per_col, vector <Cell_C*> ()));   
    for (Cell_C* cell : _vCell) {
        int row_ind = floor(cell->get_posX() / bin_width);
        int col_ind = floor(cell->get_posY() / bin_height);
        bins[row_ind][col_ind].emplace_back(cell);
    }

    vector < pair < int, pair < int, int>>> bins_size;
    for (int i=0; i<bins_per_row; ++i) {
        for (int j=0; j<bins_per_col; ++j) {
            bins_size.emplace_back(make_pair(bins[i][j].size(), make_pair(i, j)));
        }
    }
    sort(bins_size.begin(), bins_size.end());
    reverse(bins_size.begin(), bins_size.end());
    // for (int i=0; i<bin_num; i++) {
    //     cout << "bin_size = " << bins_size[i].first << "\n";
    // }
    
    double used_area[2] = {0.0, 0.0};
    double maxArea[2];
    double totalArea[2];
    for (int ind=0; ind<bin_num; ind++) {
        int i = bins_size[ind].second.first;
        int j = bins_size[ind].second.second;
        Partitioner* partitioner = new Partitioner();
        totalArea[0] = (double) _pChip->get_die(0)->get_width() * (double) _pChip->get_die(0)->get_height() * _pChip->get_die(0)->get_max_util();
        totalArea[1] = (double) _pChip->get_die(1)->get_width() * (double) _pChip->get_die(1)->get_height() * _pChip->get_die(1)->get_max_util();
        maxArea[0] = totalArea[0] - used_area[0];
        maxArea[1] = totalArea[1] - used_area[1];
        // cout << "valid area = (" << maxArea[0] << ", " << maxArea[1] << ")\n";
        // cout << "place " << used_area[0] << "," << used_area[1] << "\n";
        partitioner->parseInput(_vCell, _pChip, bins[i][j], maxArea, cutline, false);
        partitioner->initial_partition();
        if (_vCell.size() > 100000) { // case 4
        	partitioner->partition(1,2,3, true);
        } else if(_vCell.size() > 20000) { // case 3
          partitioner->partition(2,2,3, true);
        } else {
          partitioner->partition(2,2,4, true);
        }
        // partitioner->printSummary();
        vector<vector<int> >& cellPart = partitioner->get_part_result();
        
        // bool inv = (cellPart[0].size() >= cellPart[1].size()) ? false : true;
        for (int k=0; k<2; ++k){
            for(int cellId : cellPart[k]){ 
                Cell_C* cell = _vCell[cellId];
                used_area[k] += cell->get_width(_pChip->get_die(k)->get_techId()) * cell->get_height(_pChip->get_die(k)->get_techId());
                if (used_area[k] <= totalArea[k]) {
                    cell->set_die(_pChip->get_die(k));
                } else {
                    used_area[k] -= cell->get_width(_pChip->get_die(k)->get_techId()) * cell->get_height(_pChip->get_die(k)->get_techId());
                    cell->set_die(_pChip->get_die(1 - k));
                    used_area[1 - k] += cell->get_width() * cell->get_height();
                    // cout << "~~~~~~~die:" << k << ", " << totalArea[k] - used_area[k] << "\n";  
                    // cout << "die:" << 1 - k << ", " << totalArea[1 - k] - used_area[1 - k] << "\n";  
                }
                
            }
        }
    }
    _usedArea[0] = used_area[0];
    _usedArea[1] = used_area[1];



}
