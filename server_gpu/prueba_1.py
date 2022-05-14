
import rsa

pubkey, privkey = rsa.newkeys(512)
  
str1 = "I am okay"

print(type(pubkey))