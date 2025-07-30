# -*- coding: utf-8 -*-

import streamlit as st


def main():
    if st.button('Top button'):
        # 最後の試行で上のボタンがクリックされた
        st.write('Clicked')
    else:
        # クリックされなかった
        st.write('Not clicked')

    if st.button('Bottom button'):
        # 最後の試行で下のボタンがクリックされた
        st.write('Clicked')
    else:
        # クリックされなかった
        st.write('Not clicked')


if __name__ == '__main__':
    main()