#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <chrono>
#include <semaphore>


using namespace std;



class MyPrinter {
    private :
    string str; 
    int char_count;
    int thread_count;
    vector<thread> threads;
    vector<thread::id> thread_ids;
    int thread_id;
    int currently_allowed_thread;
    mutex m;
    condition_variable cv;
    int next_char; 

    public: 
        MyPrinter(string s, int c_count, int t_count) {
            str = s;
            char_count = c_count;
            thread_count = t_count;
            thread_id = 0;
            currently_allowed_thread = 0;
            next_char = 0;
        }

        int getCurrentThreadID(const thread::id& id) {
            int thread_id = 0;
            for (auto& e : thread_ids) {
                if (e == id) return thread_id;
                thread_id++; 
            }
            return -1; 
        }

        void run() {
            for (int i = 0; i < thread_count; i++) {
                thread t(&MyPrinter::printThread, this); 
                cout << "Thread: " <<t.get_id() << " is " << i << endl;
                thread_ids.push_back(t.get_id());
                threads.push_back(move(t));
            }

            for (int i = 0; i < thread_count; i++) {
                threads[i].join(); 
            }
        }

        void waitForAllThreadInit() {
            while(1) {
                if (thread_count ==thread_ids.size()) return; 
            }
        }

        void printThread() {
            while(1) {
                waitForAllThreadInit();
                std::this_thread::sleep_for(chrono::milliseconds(1000));
                unique_lock<mutex> lock(m);
                cv.wait(lock, [this] { return this_thread::get_id() == thread_ids[currently_allowed_thread];});
                print_chars();
                currently_allowed_thread++;
                if (currently_allowed_thread == thread_count) currently_allowed_thread = 0;
                if (next_char >= str.length()) next_char -= str.length();
                lock.unlock();
                cv.notify_all();
            }
        }

        void print_chars() {
            cout << "ThreadID " << getCurrentThreadID(this_thread::get_id()) << " : ";
            int printCount = 0;
            for (int i = next_char; i < str.length() && printCount < char_count; i++) {
                cout  << str[i];
                printCount++;
            }
            if (printCount < char_count) {
                for (int i = 0; i < char_count - printCount; i++) {
                    cout << str[i];
                }
            }
            next_char = next_char + char_count;
            cout << endl; 
        }
};

int main(int argc, char *argv[]) {
    if (argc != 4) {
        cout << "3 args please" << endl; 
        return 1;
    }

    string str = argv[1];
    int char_count = atoi(argv[2]);
    int thread_count = atoi(argv[3]);

    MyPrinter p(str, char_count, thread_count);
    p.run();
    return 0; 
}