#include <bits/stdc++.h>
#include <unistd.h>  
#include <string>

using namespace std;

int attackcostlist[11] = {20200, 10300, 295300, 534900, 3600, 17100, 125700, 34600, 152300, 7600};

int main(int argc, char* argv[]){
    int i = atoi(argv[1]);
    int cost = attackcostlist[i];
    pid_t pro;
    pro = fork();
    if(pro == 0){
        string command = string("python3 ./HCT.py ") + to_string(i);
        system(command.c_str());
    }
    else{
        pid_t p = fork();
        if(p==0){
            string command  = string("python3 ./de1/defend.py ") + to_string(i);
            system(command.c_str());
        }
        else{
            pid_t p1 = fork();
            if(p1 == 0){
                string command  = string("python3 ./de2/defend.py ") + to_string(i) + string(" ") + to_string(cost);
                system(command.c_str());
            }
            else{
                pid_t p2 = fork();
                if(p2 == 0){
                    string command  = string("python3 ./de3/defend.py ") + to_string(i) + string(" ") + to_string(int(cost/10));
                    system(command.c_str());
                }
                else{
                    pid_t p3 = fork();
                    if(p3 == 0){
                        string command  = string("python3 ./de4/defend.py ") + to_string(i) + string(" ") + to_string(int(cost/100));
                        system(command.c_str());
                    }
                    else{
                        string command  = string("python3 ./de5/defend.py ") + to_string(i) + string(" ") + to_string(cost);
                        system(command.c_str());
                    }
                }
            }
        }
    }

    printf("finished\n");
    return 0;
}

/*
int main(){
    for(int i=0;i<=9;++i){
        string command = string("mkdir infor") + to_string(i); 
        system(command.c_str());
        func(i);
    }
    return 0;
}
*/
