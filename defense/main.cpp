#include <bits/stdc++.h> 

using namespace std;

int main(){
    for(int i=0; i<=9; ++i){
        string command = string("mkdir ./infor") + to_string(i);
        system(command.c_str());
        command = string("./sys ") + to_string(i);
        system(command.c_str());
    }
    return 0;
}