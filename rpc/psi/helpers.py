import os
import random
import gmpy2
import grpc
import hashlib
import binascii
from Cryptodome.PublicKey import RSA
from tqdm import tqdm


def generate_rsa_keys():
    """

    :return: RSA's public key and secret key
    """
    key = RSA.generate(1024)
    pk = (key.n, key.e)
    sk = (key.n, key.d)
    return pk, sk


def hash_number(number, ret_type='int'):
    """

    :param ret_type: return type
    :param number: unhashed number
    :return:hashed number
    """
    hash_obj = hashlib.sha1(str(number).encode('utf-8'))
    digest_hex = hash_obj.hexdigest()
    if ret_type == 'int':
        return int(digest_hex, 16)
    else:
        return digest_hex


def encode_local_id_use_pk(local_ids, pk):
    """

    :param local_ids:
    :param pk: public key
    :return: encrypted id list and random number list
    """
    enc_ids = []
    ra_list = []
    for id in local_ids:
        hash_id = hash_number(id)
        hash_enc_id = hash_id % pk[0]

        # generate random number ra
        ra = int(binascii.hexlify(os.urandom(128)), 16)
        ra_enc = gmpy2.powmod(ra, pk[1], pk[0])
        ra_list.append(ra)
        enc_ids.append(int(hash_enc_id * ra_enc))  # invert mpz to int

    return enc_ids, ra_list


def decode_ids(enc_ids, sk):
    dec_ids = []
    for enc_id in tqdm(enc_ids, desc="Decode IDs"):
        dec_id = gmpy2.powmod(enc_id, sk[1], sk[0])
        dec_ids.append(int(dec_id))

    return dec_ids


def encode_and_hash_local_ids_use_sk(local_ids, sk):
    hash_enc_ids = []
    for id in tqdm(local_ids, desc='Encode & hash IDs'):
        hash_id = hash_number(id)
        hash_enc_id = gmpy2.powmod(hash_id, sk[1], sk[0])
        hash_enc_ids.append(hash_number(hash_enc_id, 'str'))

    return hash_enc_ids


def invert_and_hash_decode_ids(decode_ids, ra_list, pk):
    hash_ids = []
    for dec_id, ra in zip(decode_ids, ra_list):
        ra_inv = gmpy2.invert(ra, pk[0])
        hash_ids.append(hash_number(((dec_id * ra_inv) % pk[0]), 'str'))

    return hash_ids


def invert_and_hash_decode_ids_genexps(decode_ids, ra_list, pk):
    for hash_id_index, (dec_id, ra) in enumerate(zip(decode_ids, ra_list)):
        ra_inv = gmpy2.invert(ra, pk[0])
        yield hash_id_index, hash_number(((dec_id * ra_inv) % pk[0]), 'str')


def get_psi_index(client_hash_ids, server_hash_ids):
    psi_index = []
    for i in range(len(client_hash_ids)):
        client_hash_id = client_hash_ids[i]
        for server_hash_id in server_hash_ids:
            if client_hash_id == server_hash_id:
                psi_index.append(i)

    return psi_index


def get_psi_index_genexps(client_hash_ids_gene, server_hash_ids):
    for hash_id_index, client_hash_id in client_hash_ids_gene:
        for server_hash_id in server_hash_ids:
            if client_hash_id == server_hash_id:
                yield hash_id_index


def get_double_psi_result(local_ids, decode_ids, ra_list, pk, server_hash_ids):
    client_hash_ids = invert_and_hash_decode_ids(decode_ids, ra_list, pk)
    psi_index = get_psi_index(client_hash_ids, server_hash_ids)
    psi_result = []
    for index in psi_index:
        psi_result.append(local_ids[index])

    return psi_result


def get_double_psi_result_genexps(local_ids, decode_ids, ra_list, pk, server_hash_ids):
    # Faster
    client_hash_ids_gene = invert_and_hash_decode_ids_genexps(decode_ids, ra_list, pk)
    psi_index_gene = get_psi_index_genexps(client_hash_ids_gene, server_hash_ids)
    psi_result = []
    for index in tqdm(psi_index_gene, desc="Store TPSI Result"):
        psi_result.append(local_ids[index])

    return psi_result


def encode_empty_psi_result():
    value = random.randint(1, 10)
    length = random.randint(2, 10)
    return [value for _ in range(length)]


def get_final_psi_result(psi_dec_result):
    psi_round_result = []
    for item in psi_dec_result:
        psi_round_result.append(round(item))
    psi_result = set(psi_round_result)
    if len(psi_result) != len(psi_dec_result):
        return []

    return psi_round_result
