import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import Animated, {FadeInDown, FadeIn} from 'react-native-reanimated';
import {Ionicons} from '@expo/vector-icons';
import {useAuth} from '../context/AuthContext';
import {useServerConfig} from '../context/ServerConfigContext';
import {colors, typography, spacing, borderRadius} from '../theme';

const providers = [
  {id: 'google' as const, label: 'Google', color: '#4285F4', icon: 'logo-google' as const},
  {id: 'github' as const, label: 'GitHub', color: '#333333', icon: 'logo-github' as const},
  {id: 'discord' as const, label: 'Discord', color: '#5865F2', icon: 'logo-discord' as const},
];

export function LoginScreen() {
  const {loginWithProvider, loading} = useAuth();
  const {serverUrl, clearConfig} = useServerConfig();

  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color={colors.cyan} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Animated.View entering={FadeIn.duration(600)} style={styles.card}>
        <Animated.View entering={FadeInDown.delay(100).duration(400)}>
          <Ionicons name="trending-up" size={48} color={colors.cyan} style={styles.logo} />
          <Text style={styles.title}>Gap Fade</Text>
          <Text style={styles.subtitle}>Sign in to continue</Text>
        </Animated.View>

        <View style={styles.buttons}>
          {providers.map((p, i) => (
            <Animated.View key={p.id} entering={FadeInDown.delay(200 + i * 100).duration(300)}>
              <TouchableOpacity
                style={[styles.providerBtn, {backgroundColor: p.color}]}
                onPress={() => loginWithProvider(p.id)}
                activeOpacity={0.7}
              >
                <Ionicons name={p.icon} size={20} color="#FFFFFF" />
                <Text style={styles.providerText}>Continue with {p.label}</Text>
              </TouchableOpacity>
            </Animated.View>
          ))}
        </View>

        <Animated.View entering={FadeInDown.delay(500).duration(300)}>
          <Text style={styles.hint}>
            Opens your browser to sign in securely
          </Text>
        </Animated.View>
      </Animated.View>

      <Animated.View entering={FadeInDown.delay(600).duration(300)}>
        <Text style={styles.serverLabel}>Server: {serverUrl}</Text>
        <TouchableOpacity onPress={clearConfig} activeOpacity={0.7}>
          <Text style={styles.changeServer}>Change Server</Text>
        </TouchableOpacity>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg, justifyContent: 'center', padding: spacing.xl},
  card: {
    backgroundColor: colors.bgCard, borderRadius: borderRadius.xl,
    padding: spacing.xl, borderWidth: 1, borderColor: colors.border,
  },
  logo: {textAlign: 'center', marginBottom: spacing.md},
  title: {...typography.h1, color: colors.cyan, textAlign: 'center', marginBottom: spacing.xs},
  subtitle: {...typography.body, color: colors.textSecondary, textAlign: 'center', marginBottom: spacing.xl},
  buttons: {gap: spacing.md},
  providerBtn: {
    borderRadius: borderRadius.lg, paddingVertical: spacing.md,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.md,
  },
  providerText: {...typography.bodyMedium, color: '#FFFFFF'},
  hint: {...typography.caption, color: colors.textMuted, textAlign: 'center', marginTop: spacing.lg},
  serverLabel: {...typography.caption, color: colors.textMuted, textAlign: 'center', marginTop: spacing.xl},
  changeServer: {...typography.caption, color: colors.cyan, textAlign: 'center', marginTop: spacing.xs, paddingVertical: spacing.sm},
});
