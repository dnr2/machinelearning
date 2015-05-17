#include<bits/stdc++.h>

using namespace std;

int main(){
	string a = "abcdefg";
	string b = "ba";
	int sza = a.size();
	int szb = b.size();
	
	if( sza < szb ){
		cout << "NO" << endl;
		return 0;
	}
	
	map<char,int> mapa;
	map<char,int> mapb;
	for(int i = 0; i < szb; i++){
		mapb[b[i]]++;
	}
	int it = 0, int cont = 0;
	while( it < szb-1){
		if( mapb.count( a[it] ) > 0){
			mapa[a[it++]]++; 
			cont++;
		}
	}
	while( it < sza){
		it++;
		int begin = it - szb;
		if( mapb.count( a[it] ) > 0){
			mapa[a[it]]++; 
			cont++;
		}
		if( mapb.count( a[begin] ) > 0){
			mapa[a[begin]]--; 
			cont--;
		}
		if( cont == szb ) {
			cout << "YES"  << endl;
			return 0;
		}
	}
	cout << "NO" << endl;
	return 0;
}
