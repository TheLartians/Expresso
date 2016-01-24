

#ifndef SHA256_H
#define SHA256_H

#include <string>
#include <ostream>
#include <array>

class sha256_hash;

class SHA256{
protected:
  typedef unsigned char uint8;
  typedef unsigned int uint32;
  typedef unsigned long long uint64;
  
  const static uint32 sha256_k[];
  static const unsigned int SHA224_256_BLOCK_SIZE = (512/8);
public:
  void init();
  void update(const unsigned char *message, unsigned int len);
  void update(const std::string &str){ update((const unsigned char*)str.data(), str.size() ); }
  template <class Array> void update(const Array &arr){ update((const unsigned char*)arr.data(), arr.size()*sizeof(typename Array::value_type)/sizeof(unsigned char) ); }
  void final(unsigned char *digest);
  void final(sha256_hash & digest);
  
  static const unsigned int DIGEST_SIZE = ( 256 / 8);
  
protected:
  void transform(const unsigned char *message, unsigned int block_nb);
  unsigned int m_tot_len;
  unsigned int m_len;
  unsigned char m_block[2*SHA224_256_BLOCK_SIZE];
  uint32 m_h[8];
};

class sha256_hash:public std::array<unsigned char,SHA256::DIGEST_SIZE>{
public:
  
  sha256_hash();
  
  void invalidate();
  
  operator bool()const{
    for(auto i:*this) if(i) return true;
    return false;
  }
  
  operator std::string() const {
    return std::string(begin(),end());
  }
  
  std::string hex_string()const;
  
  size_t quick_hash = 0;
  
  bool operator==(const sha256_hash &other)const{
    if(quick_hash != other.quick_hash) return false;
    for(unsigned i = 0; i<size(); ++i) if((*this)[i] != other[i]) return false;
    return true;
  }
  
  bool operator!=(const sha256_hash &other)const{ return !operator==(other); }
  
};

std::ostream & operator<<(std::ostream & stream,const sha256_hash &hash);

namespace std{
  template<> struct hash<sha256_hash> {
    
    typedef sha256_hash argument_type;
    typedef std::size_t result_type;
    
    result_type operator()(argument_type const& s) const{
      return s.quick_hash;
    }
    
  };
}

void sha256(std::string input,sha256_hash & hash);
sha256_hash sha256(std::string input);



#endif
