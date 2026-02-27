import {TextStyle} from 'react-native';

export const fonts = {
  regular: 'Inter-Regular',
  medium: 'Inter-Medium',
  semiBold: 'Inter-SemiBold',
  bold: 'Inter-Bold',
  mono: 'JetBrainsMono-Regular',
  monoMedium: 'JetBrainsMono-Medium',
} as const;

export const typography: Record<string, TextStyle> = {
  h1: {
    fontFamily: fonts.bold,
    fontSize: 24,
    lineHeight: 32,
  },
  h2: {
    fontFamily: fonts.semiBold,
    fontSize: 20,
    lineHeight: 28,
  },
  h3: {
    fontFamily: fonts.semiBold,
    fontSize: 16,
    lineHeight: 24,
  },
  body: {
    fontFamily: fonts.regular,
    fontSize: 14,
    lineHeight: 20,
  },
  bodyMedium: {
    fontFamily: fonts.medium,
    fontSize: 14,
    lineHeight: 20,
  },
  caption: {
    fontFamily: fonts.regular,
    fontSize: 12,
    lineHeight: 16,
  },
  label: {
    fontFamily: fonts.medium,
    fontSize: 11,
    lineHeight: 14,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  mono: {
    fontFamily: fonts.mono,
    fontSize: 14,
    lineHeight: 20,
  },
  monoSmall: {
    fontFamily: fonts.mono,
    fontSize: 12,
    lineHeight: 16,
  },
  monoLarge: {
    fontFamily: fonts.monoMedium,
    fontSize: 18,
    lineHeight: 24,
  },
  heroLarge: {
    fontFamily: fonts.monoMedium,
    fontSize: 36,
    lineHeight: 44,
  },
  heroMedium: {
    fontFamily: fonts.monoMedium,
    fontSize: 28,
    lineHeight: 36,
  },
} as const;
